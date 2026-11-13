from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sherpa.cli import main
from sherpa.platform.llama_completion import ADAPTER, FRAMEWORK_URL, SUPPORTED_COMMIT, parse_completion_log, plan_completion
from sherpa.runtime.execution_contract import validate_execution
from sherpa.runtime.store import digest
from test_execution_contract import fixture, grant


def inputs():
    task = fixture(Path('/nonexistent model')).snapshot()
    task['target'] = {'kind': 'git', 'url': FRAMEWORK_URL, 'revision': 'HEAD'}
    task['contract']['execution']['framework']['commit'] = SUPPORTED_COMMIT
    task['model']['selected_set'] = 'q8'
    task['resources']['gpus'] = ['GPU-01234567-abcd-1234-abcd-0123456789ab']
    execution = task['contract']['execution']
    execution['adapter'] = {'name': ADAPTER, 'version': '1', 'build_parameters': {'cuda_arch': '120'}}
    execution['permissions']['tools'] = ['cmake', 'make', 'cc', 'cxx', 'nvcc', 'llama-completion']
    execution['acceptance'] = {'scope': 'smoke', 'parser': {'name': ADAPTER, 'version': '1'},
                               'verifier': {'name': 'gpu-smoke', 'version': '1'}, 'reference': None, 'tolerances': None}
    task['workload']['request']['parameters'] = {'seed': 0, 'max_tokens': 8, 'ctx_size': 512, 'gpu_layers': 99, 'temperature': 0}
    locations = {'run_root': '/data/run', 'source': '/data/run/workspace/framework', 'build': '/data/run/build',
                 'scratch': '/data/run/scratch', 'cmake': '/tools/cmake', 'make': '/usr/bin/make', 'cc': '/usr/bin/cc',
                 'cxx': '/usr/bin/c++', 'nvcc': '/cuda/bin/nvcc'}
    return task, locations


# Synthetic samples transcribed from the pinned source's format strings, not GPU run evidence.
LOG = '\n'.join([
    'ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16303 MiB):',
    '  Device 0: example GPU, compute capability 12.0, VMM: yes, VRAM: 16303 MiB',
    'llama_prepare_model_devices: using device CUDA0 (example GPU) (0000:01:00.0) - 15000 MiB free',
    'load_tensors: offloaded 25/25 layers to GPU',
    'llama_perf_context_print:        load time =     100.00 ms',
    'llama_perf_context_print: prompt eval time =       2.00 ms /     4 tokens (0.50 ms per token, 2000.00 tokens per second)',
    'llama_perf_context_print:        eval time =       7.00 ms /     7 runs   (1.00 ms per token, 1000.00 tokens per second)',
])


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.task, self.locations = inputs()

    def plan(self):
        return plan_completion(self.task, self.locations)

    def test_pure_planning_has_no_runtime_authority_or_io(self):
        before = copy.deepcopy((self.task, self.locations))
        with ExitStack() as stack:
            for target in ('subprocess.Popen', 'socket.socket', 'os.open', 'builtins.open', 'pathlib.Path.open',
                           'sherpa.runtime.framework.prepare', 'sherpa.platform.machine.probe', 'sherpa.agent.Router.call'):
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            result = self.plan()
        self.assertEqual(result['status'], 'draft', result['diagnostics'])
        self.assertFalse(result['executed'])
        self.assertFalse(result['execution_authorized'])
        self.assertEqual(result['readiness'], 'unverified')
        self.assertEqual((self.task, self.locations), before)
        self.assertEqual(result['task_hash'], digest(self.task))
        self.assertEqual(result['plan_hash'], digest({k: v for k, v in result.items() if k != 'plan_hash'}))
        self.assertIn('system_config_absent_or_isolated:/etc/llama.cpp/config.ini', result['required_runtime_checks'])

    def test_command_profile_matches_nonserving_completion(self):
        result = self.plan()
        configure, build, infer = result['commands']
        self.assertIn('-DLLAMA_BUILD_SERVER=OFF', configure['argv'])
        self.assertIn('-DLLAMA_USE_PREBUILT_UI=OFF', configure['argv'])
        self.assertIn('-DLLAMA_BUILD_UI=OFF', configure['argv'])
        self.assertIn('-DGGML_CUDA_CCCL_VERSION=', configure['argv'])
        self.assertIn('-DFETCHCONTENT_FULLY_DISCONNECTED=ON', configure['argv'])
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES=120', configure['argv'])
        self.assertNotIn('-DLLAMA_CURL=OFF', configure['argv'])
        self.assertEqual(build['argv'][build['argv'].index('--target') + 1], 'llama-completion')
        self.assertEqual(infer['argv'][0], '/data/run/build/bin/llama-completion')
        for opt in ('--no-conversation', '--no-display-prompt', '--offline', '--no-warmup', '--no-context-shift'):
            self.assertIn(opt, infer['argv'])
        self.assertNotIn('--no-mmap', infer['argv'])
        self.assertEqual(infer['argv'][infer['argv'].index('--load-mode') + 1], 'none')
        self.assertEqual(infer['argv'][infer['argv'].index('--device') + 1], 'CUDA0')
        self.assertEqual(infer['argv'][infer['argv'].index('--fit') + 1], 'off')
        self.assertTrue(all(c['stdin'] == 'DEVNULL' for c in result['commands']))

    def test_q8_selected_q4_not_implicitly_substituted(self):
        result = self.plan()
        argv = result['commands'][-1]['argv']
        self.assertEqual(argv[argv.index('-m') + 1], '/nonexistent model/alternative.gguf')
        self.task['model']['selected_set'] = None
        self.assertEqual(self.plan()['status'], 'invalid')

    def test_prompt_is_single_argv_data_not_shell_code(self):
        prompt = 'hello; $(touch unexpected) "quoted"\n--model-url https://bad.example/model'
        self.task['workload']['request']['input'] = prompt
        result = self.plan()
        argv = result['commands'][-1]['argv']
        self.assertEqual(argv[argv.index('-p') + 1], prompt)
        self.assertEqual(argv.count(prompt), 1)
        self.assertNotIn('--model-url', argv)
        self.assertFalse(result['execution_authorized'])

    def test_environment_is_explicit_and_not_inherited(self):
        with patch.dict(os.environ, {'LLAMA_ARG_MODEL_URL': 'https://bad.example', 'HF_TOKEN': 'test-secret', 'LD_PRELOAD': '/tmp/evil.so'}):
            result = self.plan()
        for c in result['commands']:
            self.assertFalse(c['inherit_environment'])
            self.assertEqual(c['env']['CUDA_VISIBLE_DEVICES'], self.task['resources']['gpus'][0])
            self.assertFalse({'LLAMA_ARG_MODEL_URL', 'HF_TOKEN', 'LD_PRELOAD'} & c['env'].keys())
            self.assertTrue(c['env']['HOME'].startswith(self.locations['scratch']))

    def test_wrong_framework_revision_or_adapter_is_rejected(self):
        for path, value in ((('target', 'url'), 'https://example.com/other.git'),
                            (('contract', 'execution', 'framework', 'commit'), 'a' * 40),
                            (('contract', 'execution', 'adapter', 'name'), 'llama-cli')):
            task = copy.deepcopy(self.task)
            node = task
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            self.assertEqual(plan_completion(task, self.locations)['status'], 'invalid')

    def test_paths_cannot_escape_or_overlap(self):
        for key, value in (('build', '/data/run/workspace/framework/build'), ('scratch', '/data/run/build/tmp'),
                           ('source', '/other/source'), ('run_root', '/'), ('cmake', '/data/run/build/cmake'),
                           ('nvcc', '/cuda/bin/nvcc;-DOPTION=ON'), ('build', '/data/run/../out'), ('source', 'relative'),
                           ('cmake', '//data/run/build/cmake'), ('build', '//data/run/build')):
            with self.subTest(key=key, value=value):
                loc = {**self.locations, key: value}
                self.assertEqual(plan_completion(self.task, loc)['status'], 'invalid')
        for root in ('/data/run/model', '//data/run/model'):
            self.task['model']['path'] = root
            result = self.plan()
            self.assertEqual(result['status'], 'invalid')
            self.assertEqual(result['commands'], [])

    def test_no_implicit_arch_or_unbounded_generation(self):
        for name, value in (('max_tokens', -1), ('max_tokens', True), ('gpu_layers', 'all'), ('gpu_layers', 0),
                            ('ctx_size', 8), ('temperature', 0.5), ('seed', 4294967295)):
            t = copy.deepcopy(self.task)
            t['workload']['request']['parameters'][name] = value
            self.assertEqual(plan_completion(t, self.locations)['status'], 'invalid')
        for arch in ('native', '120;80', None, 120):
            self.task['contract']['execution']['adapter']['build_parameters']['cuda_arch'] = arch
            self.assertEqual(self.plan()['status'], 'invalid')

    def test_no_silent_shard_or_probability_support(self):
        self.task['model']['sets'][1]['support'] = [{'path': 'tokenizer.gguf', 'sha256': 'a' * 64, 'bytes': 4}]
        self.assertEqual(self.plan()['status'], 'invalid')
        self.task, _ = inputs()
        self.task['contract']['execution']['acceptance']['scope'] = 'behavior'
        self.assertEqual(self.plan()['status'], 'invalid')

    def test_missing_fields_and_bad_shapes_return_errors(self):
        self.locations.pop('nvcc')
        self.task['workload']['request']['parameters']['max_tokens'] = -1
        result = self.plan()
        self.assertEqual(result['commands'], [])
        self.assertTrue(any(d['field'] == 'locations.nvcc' for d in result['diagnostics']))
        self.assertTrue(any(d['field'] == 'request.parameters.max_tokens' for d in result['diagnostics']))
        for key in ('contract', 'workload', 'model', 'resources'):
            for value in (None, [], 4, 'wrong'):
                t, loc = inputs()
                t[key] = value
                self.assertEqual(plan_completion(t, loc)['status'], 'invalid')

    def test_e1_can_check_same_task_without_becoming_execution_permission(self):
        with tempfile.TemporaryDirectory() as temp:
            self.task['model']['path'] = temp
            (Path(temp) / 'alternative.gguf').write_bytes(b'12345678')
            report = validate_execution(self.task, {'kind': 'run_once', 'build_call': 'call-build'}, grant(self.task))
            self.assertEqual(report['status'], 'valid', report['diagnostics'])
            self.assertEqual(self.plan()['status'], 'draft')
            self.assertFalse(report['execution_authorized'])

    def test_cli_draft_creates_no_store_or_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'task.json').write_text(json.dumps(self.task))
            (root / 'locations.json').write_text(json.dumps(self.locations))
            out = io.StringIO()
            with patch('sherpa.cli.Store', side_effect=AssertionError('no store')), patch('subprocess.Popen', side_effect=AssertionError('no process')), redirect_stdout(out):
                code = main(['env', 'plan-completion', str(root / 'task.json'), '--locations', str(root / 'locations.json')])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'draft')
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['locations.json', 'task.json'])


class LogEvidenceTests(unittest.TestCase):
    def test_complete_sample_extracts_evidence_not_verdict(self):
        result = parse_completion_log(LOG, complete=True)
        self.assertEqual(result['status'], 'evidence_present', result['issues'])
        self.assertEqual(result['device']['backend'], 'CUDA0')
        self.assertEqual(result['device']['bus_id'], '0000:01:00.0')
        self.assertEqual(result['offloaded_layers'], 25)
        self.assertEqual(result['prompt_tokens'], 4)
        self.assertEqual(result['eval_runs'], 7)
        self.assertIsNone(result['generated_tokens'])
        self.assertFalse(result['runtime_verified'])
        self.assertFalse(result['behavior_verified'])
        self.assertEqual([e['line'] for e in result['evidence']], [3, 4, 5, 6, 7])

    def test_enumeration_or_offload_alone_cannot_pass(self):
        for sample in ('\n'.join(LOG.splitlines()[:2]), LOG.splitlines()[3], ''):
            result = parse_completion_log(sample, complete=True)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertIsNone(result['eval_runs'])
        self.assertIsNone(parse_completion_log('\n'.join(LOG.splitlines()[:2]), complete=True)['device'])

    def test_truncated_log_and_duplicate_runs_are_ambiguous(self):
        self.assertIn('partial_log', parse_completion_log(LOG, complete=False)['issues'])
        result = parse_completion_log(LOG + '\n' + LOG, complete=True)
        self.assertEqual(result['status'], 'inconclusive')
        self.assertIn('ambiguous_repeated_device', result['issues'])
        self.assertIn('ambiguous_repeated_eval', result['issues'])

    def test_zero_offload_bad_counts_and_missing_decode_are_not_positive_evidence(self):
        for old, new in (('25/25', '0/25'), ('25/25', '26/25'), ('7 runs', '0 runs')):
            with self.subTest(new=new):
                self.assertEqual(parse_completion_log(LOG.replace(old, new), complete=True)['status'], 'inconclusive')
        result = parse_completion_log('\n'.join(LOG.splitlines()[:-1]), complete=True)
        self.assertIsNone(result['eval_runs'])
        self.assertEqual(result['prompt_tokens'], 4)

    def test_server_and_prompt_text_are_not_completion_metrics(self):
        sample = 'slot print_timing: id 0 | task 1 | eval time = 7.0 ms / 7 tokens\n' + 'model reply: ' + LOG.splitlines()[3]
        result = parse_completion_log(sample, complete=True)
        self.assertIsNone(result['eval_runs'])
        self.assertIsNone(result['offloaded_layers'])

    def test_bad_or_extreme_logs_are_bounded(self):
        for sample, complete in ((None, True), ('x' * (1024 * 1024 + 1), True), ('文' * 400000, True),
                                 ('\ud800', True), (LOG, 'true'), ('load_tensors: offloaded ' + '9' * 5000 + '/25 layers to GPU', True)):
            result = parse_completion_log(sample, complete=complete)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertTrue(result['issues'])
            self.assertLess(len(json.dumps(result)), 5000)


if __name__ == '__main__':
    unittest.main()
