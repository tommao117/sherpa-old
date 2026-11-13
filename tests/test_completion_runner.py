from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from sherpa.cli import main
from sherpa.runtime.calls import Calls, RunStopped
from sherpa.runtime.completion import CompletionRunner, completion_template, run_completion, stage_completion
from sherpa.runtime.completion_host import CompletionHost, gpu_snapshot
from sherpa.runtime.execution_contract import load_document
from sherpa.runtime.managed_process import run_process
from sherpa.runtime.store import Store, atomic_json, digest
from test_llama_completion import LOG

UUID = 'GPU-01234567-abcd-1234-abcd-0123456789ab'


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sherpa-live-offline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'data')
        self.input = completion_template()
        self.input['pinned_source'] = str(self.root / 'source')
        Path(self.input['pinned_source']).mkdir()
        git = Path(self.input['pinned_source']) / '.git'
        git.mkdir()
        (git / 'config').write_text('[core]\nrepositoryformatversion = 0\nbare = false\n')
        from sherpa.runtime import framework
        self.real_prepare, self.real_verify = framework.prepare, framework.verify_clean
        model = self.root / 'model'
        model.mkdir()
        (model / 'jevos-v4-q8_0.gguf').write_bytes(b'fixture weight, not a real model')
        t = self.input['task']
        t['model']['path'] = str(model)
        t['resources']['gpus'] = [UUID]
        t['contract']['execution']['adapter']['build_parameters']['cuda_arch'] = '120'
        t['budget']['seconds'] = 60
        t['contract']['execution']['limits'].update(step_seconds=10, cpu_threads=1, memory_mib=512, gpu_memory_mib=64, disk_mib=512, output_mib=1)
        binary_code = '#!' + sys.executable + '\nimport sys\nprint("response")\nprint(' + repr(LOG) + ',file=sys.stderr)\n'
        tool = self.root / 'fake-tool'
        tool.write_text('#!' + sys.executable + '\nimport sys,pathlib\na=sys.argv[1:]\n'
                        'if a and a[0].startswith("--query-gpu"):\n print(' + repr(UUID + ',00000000:01:00.0,0,16000,0') + ')\n'
                        'elif a and a[0].startswith("--query-compute-apps"):\n pass\n'
                        'elif "-B" in a:\n p=pathlib.Path(a[a.index("-B")+1]); p.mkdir(); (p/"CMakeCache.txt").write_text("fake cache")\n'
                        'elif "--build" in a:\n p=pathlib.Path(a[a.index("--build")+1])/"bin"; p.mkdir(); b=p/"llama-completion"; b.write_text(' + repr(binary_code) + '); b.chmod(0o700)\n')
        tool.chmod(0o700)
        self.input['tools'] = {name: str(tool) for name in self.input['tools']}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('sherpa.runtime.completion.framework.prepare', side_effect=self.fake_prepare))
        self.stack.enter_context(patch('sherpa.runtime.completion.framework.verify_clean'))
        self.stack.enter_context(patch.object(CompletionHost, 'check_config'))
        self.stack.enter_context(patch('socket.socket', side_effect=AssertionError('network forbidden')))
        self.stack.enter_context(patch('sherpa.agent.Router.call', side_effect=AssertionError('provider forbidden')))

    def fake_prepare(self, run, calls, *, max_mib, pinned):
        started = time.monotonic()
        rec, directory = calls.new('tool', 'framework.prepare', {'target': run.state()['task']['target'], 'max_mib': max_mib, 'pinned_commit': pinned['commit']})
        rec.update(synthetic=False, tool_version='framework-import-v2')
        source = run.path / 'workspace/framework'
        source.mkdir(parents=True)
        (source / 'CMakeLists.txt').write_text('offline fixture')
        result = {'commit': pinned['commit'], 'source_commit': pinned['commit'], 'import_mode': 'pinned_local_copy',
                  'source': run.state()['task']['target'], 'path': str(source), 'framework_max_mib': max_mib,
                  'import_version': 'framework-import-v2', 'model_weights_copied': False, 'submodules_not_fetched': []}
        calls.finish(rec, directory, 'succeeded', result, started)
        run.update('framework_pinned', framework=result)
        return result

    def stage(self):
        return stage_completion(self.store, self.input['task'], self.input['pinned_source'], self.input['tools'], allow_input_preparation=True)

    def execute(self, run):
        return run_completion(self.store, run.id, run.state()['ticket_hash'])

    def test_full_staging_approval_and_runtime_pipeline(self):
        run = self.stage()
        self.assertEqual(run.state()['phase'], 'awaiting_approval')
        self.assertEqual(len(run.records('calls')), 2)
        self.assertFalse((run.path / 'build').exists())
        result = self.execute(run).state()
        self.assertEqual(result['status'], 'completed', result['stop_reason'])
        self.assertEqual(result['result_status'], 'gpu_smoke_passed')
        self.assertFalse(result['behavior_verified'])
        self.assertEqual(result['cost']['calls'], 6)
        records = run.records('calls')
        self.assertEqual([r['name'] for r in records][-4:], ['completion.preflight', 'completion.configure', 'completion.build', 'completion.run_once'])
        for r in records[-3:]:
            self.assertTrue(r['cleanup_complete'])
            self.assertTrue(Calls.group_gone(r['process']['pid']))
        with self.assertRaisesRegex(ValueError, 'no replay'):
            self.execute(run)

    def test_no_staging_io_without_explicit_permission(self):
        with patch('sherpa.runtime.completion.model_identity', side_effect=AssertionError('no weight read')):
            with self.assertRaisesRegex(ValueError, 'allow-input-preparation'):
                stage_completion(self.store, self.input['task'], self.input['pinned_source'], self.input['tools'])
        self.assertFalse(self.store.root.exists())

    def test_no_gpu_probe_during_staging(self):
        with patch.object(CompletionHost, 'sample', side_effect=AssertionError('no GPU probe')):
            run = self.stage()
        self.assertEqual(run.state()['status'], 'created')
        self.assertEqual(self.input['task']['model']['sets'][0]['weights'][0]['sha256'], None)

    def test_approval_mismatch_and_ticket_tamper_cannot_start(self):
        run = self.stage()
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'approval does not match'):
                run_completion(self.store, run.id, '0' * 64)
            path = run.path / 'completion-ticket.json'
            ticket = load_document(path)
            ticket['tools']['cmake']['sha256'] = '0' * 64
            atomic_json(path, ticket)
            with self.assertRaisesRegex(ValueError, 'approval does not match'):
                self.execute(run)
        self.assertEqual(run.state()['status'], 'created')

    def test_weight_and_tool_changes_rejected_before_gpu(self):
        for what in ('weight', 'tool'):
            with self.subTest(what=what):
                run = self.stage()
                p = Path(self.input['task']['model']['path']) / 'jevos-v4-q8_0.gguf' if what == 'weight' else Path(self.input['tools']['cmake'])
                original = p.read_bytes()
                p.write_bytes(original + b' ')
                try:
                    with patch.object(CompletionHost, 'sample', side_effect=AssertionError('no GPU probe')):
                        state = self.execute(run).state()
                    self.assertEqual(state['status'], 'failed')
                    self.assertIn('changed', state['stop_reason'])
                finally:
                    p.write_bytes(original)

    def test_busy_unknown_or_memory_limited_gpu_blocks_commands(self):
        for error in (RunStopped('failed', 'GPU busy'), ValueError('GPU process visibility unavailable'), RunStopped('budget_exhausted', 'GPU memory budget exceeded')):
            run = self.stage()
            with patch.object(CompletionHost, 'sample', side_effect=error):
                state = self.execute(run).state()
            self.assertIn(state['status'], ('failed', 'budget_exhausted'))
            self.assertFalse((run.path / 'build').exists())
            self.assertEqual(len(run.records('calls')), 3)

    def test_same_size_model_replaced_during_build_never_runs_inference(self):
        run = self.stage()
        original = CompletionRunner.execute_phase
        invoked = []
        real_process = run_process
        def execute(runner, index):
            result = original(runner, index)
            if index == 1:
                p = Path(self.input['task']['model']['path']) / 'jevos-v4-q8_0.gguf'
                p.write_bytes(b'x' * p.stat().st_size)
            return result
        def process(command, *args, **kwargs):
            invoked.append(command.get('phase'))
            return real_process(command, *args, **kwargs)
        with patch.object(CompletionRunner, 'execute_phase', execute), patch('sherpa.runtime.completion.run_process', process):
            state = self.execute(run).state()
        self.assertEqual(state['status'], 'failed')
        self.assertIn('model changed before inference', state['stop_reason'])
        self.assertNotIn('run_once', invoked)

    def test_hash_guard_does_not_walk_disk_for_every_block(self):
        run = self.stage()
        ticket = load_document(run.path / 'completion-ticket.json')
        runner = CompletionRunner(run, ticket)
        with patch('sherpa.runtime.completion.disk_bytes', return_value=0) as disk, patch.object(runner.calls, 'check_interrupt') as cancelled:
            for _ in range(300):
                runner.guard()
            self.assertEqual(cancelled.call_count, 300)
            self.assertEqual(disk.call_count, 1)
            runner.guard(force=True)
            self.assertEqual(disk.call_count, 2)

    def test_failed_inference_preserves_actual_process_and_ends_run(self):
        tool = Path(self.input['tools']['cmake'])
        tool.write_text(tool.read_text().replace('print("response")', 'print("")'))
        run = self.stage()
        state = self.execute(run).state()
        self.assertEqual(state['status'], 'failed')
        self.assertFalse(state['behavior_verified'])
        record = run.records('calls')[-1]
        self.assertEqual(record['name'], 'completion.run_once')
        result = load_document(run.path / record['output'])
        self.assertEqual(result['process']['exit_code'], 0)
        self.assertEqual(result['verification']['status'], 'inconclusive')
        self.assertTrue(record['cleanup_complete'])

    def test_real_pinned_prepare_without_fsmonitor_execution_or_gpu(self):
        from sherpa.runtime import workspace
        source = Path(self.input['pinned_source'])
        workspace.git(source, 'init')
        (source / 'CMakeLists.txt').write_text('tiny source fixture')
        commit = workspace.commit_all(source, 'test fixture')
        hook = self.root / 'fsmonitor-hook'
        marker = self.root / 'unexpected-hook'
        hook.write_text('#!/bin/sh\ntouch ' + str(marker) + '\n')
        hook.chmod(0o700)
        config = source / '.git/config'
        config.write_text(config.read_text() + '\n[core]\nfsmonitor = ' + str(hook) + '\n')
        before = config.read_bytes()
        task = self.input['task']
        task['target']['revision'] = commit
        task['contract']['execution']['framework']['commit'] = commit
        with patch('sherpa.runtime.completion.SUPPORTED_COMMIT', commit), patch('sherpa.platform.llama_completion.SUPPORTED_COMMIT', commit), \
             patch('sherpa.runtime.completion.framework.prepare', self.real_prepare), patch('sherpa.runtime.completion.framework.verify_clean', self.real_verify), \
             patch.object(CompletionHost, 'sample', side_effect=AssertionError('no GPU')):
            run = self.stage()
        self.assertFalse(marker.exists())
        self.assertEqual(config.read_bytes(), before)
        self.assertEqual(run.state()['phase'], 'awaiting_approval')
        self.assertEqual(run.state()['framework']['import_mode'], 'pinned_local_copy')
        self.assertEqual((run.path / 'workspace/framework/CMakeLists.txt').read_text(), 'tiny source fixture')
        git_requests = [load_document(p)['args'] for p in run.path.glob('calls/*/git-*/request.json')]
        self.assertTrue(git_requests)
        self.assertFalse(any(args[0] == 'fetch' for args in git_requests))

    def test_source_git_filters_are_rejected_before_any_command(self):
        config = Path(self.input['pinned_source']) / '.git/config'
        config.write_text(config.read_text() + '\n[filter "custom"]\nclean = do-not-run\n')
        with patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'custom Git filters'):
                self.stage()

    def test_bom_does_not_hide_git_config_includes_or_filters(self):
        config = Path(self.input['pinned_source']) / '.git/config'
        for section in ('[include]\npath=/outside', '[filter "external"]\nclean=do-not-run'):
            config.write_bytes(b'\xef\xbb\xbf' + section.encode())
            with self.subTest(section=section), patch('subprocess.Popen', side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError, 'custom Git filters'):
                    self.stage()

    def test_git_runtime_identity_change_invalidates_ticket_before_gpu(self):
        run = self.stage()
        ticket = load_document(run.path / 'completion-ticket.json')
        self.assertTrue({'workspace.py', 'git_worker.py', 'calls.py', 'store.py', 'models.py'} <= ticket['controller'].keys())
        changed = {**ticket['controller'], 'workspace.py': '0' * 64}
        with patch('sherpa.runtime.completion.controller_identity', return_value=changed), patch.object(CompletionHost, 'sample', side_effect=AssertionError('no GPU')):
            state = self.execute(run).state()
        self.assertEqual(state['status'], 'failed')
        self.assertIn('controller changed', state['stop_reason'])

    def test_malformed_staging_input_returns_actionable_error_without_io(self):
        for value in ({}, [], {'model': None}, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stage_completion(self.store, value, self.input['pinned_source'], self.input['tools'], allow_input_preparation=True)
        self.assertFalse(self.store.root.exists())

    def test_config_presence_blocks_staging_without_execution(self):
        with patch.object(CompletionHost, 'check_config', side_effect=ValueError('implicit config')):
            with self.assertRaisesRegex(ValueError, 'implicit config'):
                self.stage()
        self.assertTrue(all(s['status'] == 'failed' for s in self.store.runs()))

    def test_prepare_failure_is_recorded_and_never_creates_ticket(self):
        with patch('sherpa.runtime.completion.framework.prepare', side_effect=RunStopped('timed_out', 'prepare timeout')):
            with self.assertRaises(RunStopped):
                self.stage()
        state = self.store.runs()[0]
        self.assertEqual(state['status'], 'timed_out')
        self.assertFalse((self.store.open_run(state['id']).path / 'completion-ticket.json').exists())

    def test_cancel_waiting_for_approval_stops_before_preflight(self):
        run = self.stage()
        run.cancel('wait until GPU is free')
        with patch.object(CompletionHost, 'sample', side_effect=AssertionError('no GPU')):
            state = self.execute(run).state()
        self.assertEqual(state['status'], 'cancelled')
        self.assertEqual(len(run.records('calls')), 2)

    def test_cpu_or_unbounded_task_rejected_before_staging(self):
        self.input['task']['budget']['seconds'] = 100000
        with self.assertRaises(ValueError):
            self.stage()
        self.assertFalse(self.store.root.exists())

    def test_cli_template_and_two_step_flow(self):
        out = io.StringIO()
        with redirect_stdout(out), patch('subprocess.Popen', side_effect=AssertionError('template must not execute')):
            self.assertEqual(main(['env', 'completion-template']), 0)
        self.assertIsNone(json.loads(out.getvalue())['task']['model']['sets'][0]['weights'][0]['sha256'])
        inp = self.root / 'input.json'
        atomic_json(inp, self.input)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(['--data-root', str(self.store.root), 'env', 'stage-completion', str(inp), '--allow-input-preparation']), 0)
        stage = json.loads(out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(['--data-root', str(self.store.root), 'env', 'run-completion', stage['run_id'], '--approve-ticket-hash', stage['ticket_hash']]), 0, out.getvalue())
        self.assertEqual(json.loads(out.getvalue())['result_status'], 'gpu_smoke_passed')


class GPUParsingTests(unittest.TestCase):
    def snapshot(self, row=None, apps='', own=(), memory=64):
        return gpu_snapshot(row or f'{UUID},00000000:01:00.0,0,16000,0', apps, UUID, own_pids=own, memory_mib=memory)

    def test_idle_identity_and_own_process(self):
        self.assertEqual(self.snapshot()['bus_id'], '0000:01:00.0')
        result = self.snapshot(f'{UUID},0000:01:00.0,40,16000,95', f'{UUID},123', (123,))
        self.assertEqual(result['processes'], [123])

    def test_unknown_busy_and_over_budget_rejected(self):
        for row, apps in ((f'{UUID},0000:01:00.0,N/A,16000,0', ''),
                          (f'{UUID},0000:01:00.0,0,16000,5', ''),
                          (f'{UUID},0000:01:00.0,0,16000,0', f'{UUID},123'),
                          (f'{UUID},0000:01:00.0,100,16000,0', ''),
                          (f'{UUID},unknown,0,16000,0', ''),
                          (f'{UUID},0000:01:00.0,0,16000,0', f'{UUID},[N/A]')):
            with self.subTest(row=row), self.assertRaises((ValueError, RunStopped)):
                self.snapshot(row, apps)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sherpa-process-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = 0

    def run_code(self, code, *, timeout=3, output=1, check=lambda: None, start=lambda p: None, monitor=lambda g, poll: None):
        self.index += 1
        cmd = {'argv': [sys.executable, '-I', '-B', '-c', code], 'cwd': str(self.root), 'env': {'PATH': os.defpath},
               'inherit_environment': False, 'stdin': 'DEVNULL',
               'limits': {'step_seconds': timeout, 'cpu_threads': 1, 'memory_mib': 128, 'disk_mib': 1, 'output_mib': output}}
        return run_process(cmd, self.root / str(self.index), check, on_start=start, monitor=monitor)

    def test_success_failure_and_no_children_remaining(self):
        for code, expected in (('print("ok")', 0), ('import sys; sys.exit(7)', 7)):
            result = self.run_code(code)
            self.assertEqual(result['exit_code'], expected)
            self.assertTrue(result['cleanup_complete'])
            self.assertTrue(Calls.group_gone(result['group']))

    def test_timeout_output_cap_and_descendant_cleanup(self):
        for code, timeout, output in (('import time; time.sleep(10)', .2, 1),
                                     ('import sys; sys.stderr.write("x"*100000)', 2, .01),
                                     ('import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"])', 2, 1)):
            result = self.run_code(code, timeout=timeout, output=output)
            self.assertTrue(result['cleanup_complete'], result)
            self.assertTrue(Calls.group_gone(result['group']))
            self.assertLessEqual(sum(result['captured_bytes'].values()), int(output * 1024 * 1024))

    def test_cancel_and_cleanup_error_are_terminal(self):
        count = 0
        def check():
            nonlocal count
            count += 1
            if count > 3:
                raise RunStopped('cancelled', 'test')
        result = self.run_code('import time;time.sleep(10)', check=check)
        self.assertEqual(result['status'], 'cancelled')
        self.assertTrue(Calls.group_gone(result['group']))
        with patch.object(Calls, 'cleanup', side_effect=OSError('injected')):
            result = self.run_code('print("done")')
        self.assertEqual(result['status'], 'cleanup_failed')
        self.assertTrue(Calls.group_gone(result['group']))

    def test_slow_monitor_does_not_hide_step_timeout(self):
        result = self.run_code('import time; time.sleep(.2)', timeout=.15,
                               monitor=lambda g, poll: time.sleep(.3))
        self.assertEqual(result['status'], 'timed_out', result)
        self.assertTrue(Calls.group_gone(result['group']))

    def test_monitor_wait_checks_outer_rss_and_cancellation(self):
        def monitor(group, poll):
            with patch('sherpa.runtime.managed_process.group_rss_mib', return_value=129):
                poll()
        result = self.run_code('import time; time.sleep(10)', monitor=monitor)
        self.assertEqual(result['status'], 'budget_exhausted', result)
        self.assertIn('memory', result['reason'])
        self.assertTrue(Calls.group_gone(result['group']))

    def test_parent_death_after_worker_exit_cleans_held_pipes(self):
        helper = self.root / 'parent.py'
        package = str(Path(__file__).resolve().parents[1])
        helper.write_text('import sys,time, pathlib\nsys.path.insert(0,' + repr(package) + ')\n'
                          'from sherpa.runtime.managed_process import run_process\n'
                          'from sherpa.runtime.store import atomic_json\n'
                          'root=pathlib.Path(' + repr(str(self.root)) + ')\n'
                          'cmd={"argv":[sys.executable,"-c","import subprocess,sys; subprocess.Popen([sys.executable,\'-c\',\'import time;time.sleep(30)\'])"],"cwd":str(root),"env":{},"inherit_environment":False,"stdin":"DEVNULL","limits":{"step_seconds":10,"cpu_threads":1,"memory_mib":128,"disk_mib":1,"output_mib":1}}\n'
                          'def monitor(g, poll):\n if (root/"proc/child.json").exists() and "exit_code" in (root/"proc/child.json").read_text() and "null" not in (root/"proc/child.json").read_text():\n  (root/"ready").write_text(str(g)); time.sleep(30)\n'
                          'run_process(cmd,root/"proc",lambda:None,monitor=monitor)\n')
        proc = subprocess.Popen([sys.executable, '-B', str(helper)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        group = None
        try:
            deadline = time.monotonic() + 6
            while not (self.root / 'ready').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((self.root / 'ready').exists())
            group = int((self.root / 'ready').read_text())
            proc.kill()
            proc.communicate(timeout=3)
            deadline = time.monotonic() + 3
            while not Calls.group_gone(group) and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(Calls.group_gone(group))
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=3)
            if group and not Calls.group_gone(group):
                os.killpg(group, signal.SIGKILL)


if __name__ == '__main__':
    unittest.main()
