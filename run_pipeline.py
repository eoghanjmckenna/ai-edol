#!/usr/bin/env python3
"""
AI-EDOL pipeline runner.

Runs the synthetic building-energy-data pipeline end to end from a single
flat YAML config (see config.yaml):

    dummy_data -> tokeniser -> sharding -> training -> inference -> evaluation

Each stage reads its own section of the config. All data paths are resolved
relative to the run directory (experiment.output_dir).

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --stages tokeniser sharding
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile
import pickle
from pathlib import Path

import yaml
import torch
import pandas as pd

# Make the src/ packages importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.gpt_model import GPTSmartMeterModel
from preprocessing.sharding import DataSharder
from training.trainer import ModelTrainer, create_optimizer_and_loss
from training.training_utils import set_seed, save_experiment_config
from inference.generator import SyntheticDataGenerator
from evaluation.mapreduce import ParallelEvaluator, visualize_results
from utils.universal_tokeniser_naive import load_tokenisers
from utils.json_logger import setup_json_logging
from utils.device_utils import get_device_from_config

VALID_STAGES = ['dummy_data', 'tokeniser', 'sharding', 'training', 'inference', 'evaluation']


def log(message, stage='PIPELINE'):
    """Print a stage-prefixed log line."""
    print(f"[{stage}] {message}", flush=True)


def resolve(run_path, path):
    """Resolve a path relative to run_path; absolute paths pass through."""
    return os.path.join(run_path, path)


# ===========================================================================
# Stage: dummy data
# ===========================================================================
def stage_dummy_data(config, run_path):
    """Generate SERL-like dummy data by running the dummy data generator."""
    log("Generating dummy data...", 'DUMMY_DATA')
    dd_config = config.get('dummy_data', {}).copy()
    script = os.path.join(SCRIPT_DIR, 'src', 'utils', 'generate_dummy_data.py')

    # Resolve the output folder into the run directory before generation
    if 'output' in dd_config and 'folder' in dd_config['output']:
        out_folder = resolve(run_path, dd_config['output']['folder'])
        dd_config['output']['folder'] = out_folder
        os.makedirs(out_folder, exist_ok=True)
        log(f"Output directory: {out_folder}", 'DUMMY_DATA')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'dummy_data': dd_config}, f, default_flow_style=False)
        temp_config = f.name
    try:
        result = subprocess.run(
            [sys.executable, script, '--config', temp_config],
            cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=900,
        )
        if result.returncode != 0:
            log(f"Dummy data generation failed:\n{result.stderr}", 'DUMMY_DATA')
            return False
        log("Dummy data generated successfully", 'DUMMY_DATA')
        return True
    finally:
        os.unlink(temp_config)


# ===========================================================================
# Stage: tokeniser
# ===========================================================================
def stage_tokeniser(config, run_path):
    """Build tokenisers and tokenise the data by running the tokeniser script."""
    log("Running tokenisation...", 'TOKENISER')
    tk_config = config.get('tokeniser', {}).copy()
    script = os.path.join(SCRIPT_DIR, 'src', 'utils', 'universal_tokeniser_naive.py')

    if 'output_folder' in tk_config:
        out = resolve(run_path, tk_config['output_folder'])
        tk_config['output_folder'] = out
        os.makedirs(out, exist_ok=True)
    if 'data_folder' in tk_config:
        tk_config['data_folder'] = resolve(run_path, tk_config['data_folder'])
    if 'save_tokenisers' in tk_config:
        save_tok = resolve(run_path, tk_config['save_tokenisers'])
        tk_config['save_tokenisers'] = save_tok
        os.makedirs(os.path.dirname(save_tok), exist_ok=True)

    filtering = tk_config.get('filtering', {})
    if filtering.get('enabled', False):
        for key in ['path_to_list_of_exporter_puprns', 'path_to_participant_summary']:
            if key in filtering:
                tk_config['filtering'][key] = resolve(run_path, filtering[key])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'tokeniser': tk_config}, f, default_flow_style=False)
        temp_config = f.name
    try:
        result = subprocess.run(
            [sys.executable, script, '--config', temp_config],
            cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=1800,
        )
        if result.stdout:
            log(f"Output:\n{result.stdout}", 'TOKENISER')
        if result.returncode != 0:
            log(f"Tokenisation failed:\n{result.stderr}", 'TOKENISER')
            return False
        log("Tokenisation completed successfully", 'TOKENISER')
        return True
    finally:
        os.unlink(temp_config)


# ===========================================================================
# Stage: sharding
# ===========================================================================
def stage_sharding(config, run_path):
    """Create train/val/test shards from the tokenised data."""
    log("Sharding data...", 'SHARDING')
    sharding_logger = setup_json_logging("sharding_log", run_path)
    sharding_logger.log_stage_start('sharding')
    sharder = DataSharder(config, run_path, logger=sharding_logger)
    sharder.shard_data()
    sharding_logger.log_stage_end('sharding')
    log("Sharding completed successfully", 'SHARDING')
    return True


# ===========================================================================
# Training helpers
# ===========================================================================
def _load_tokenizers_and_vocab(tokeniser_file):
    """Load tokenisers and compute per-feature vocabulary sizes."""
    (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
     elec_bin_means, gas_bin_means, value_transform) = load_tokenisers(tokeniser_file)
    return {
        'elec_boundaries': elec_boundaries,
        'gas_boundaries': gas_boundaries,
        'temp_boundaries': temp_boundaries,
        'solar_boundaries': solar_boundaries,
        'elec_bin_means': elec_bin_means,
        'gas_bin_means': gas_bin_means,
        'value_transform': value_transform,
    }


def _create_model(training_config, vocab_sizes, output_variable):
    """Create a single-output GPT model for electricity or gas."""
    model_config = training_config['model'][output_variable]
    context_length = training_config['model']['context_length']
    return GPTSmartMeterModel(
        vocab_sizes=vocab_sizes,
        d_model=model_config['n_embd'],
        nhead=model_config['n_head'],
        num_layers=model_config['n_layer_do'],
        dim_feedforward=model_config['n_embd'] * 2,
        context_length=context_length,
        dropout=model_config['dropout'],
        output_variable=output_variable,
    )


def _train_one(output_variable, training_config, run_path, vocab_sizes, bin_means,
               train_shards_dir, logger):
    """Create, train and clean up one single-output model."""
    log(f"Starting {output_variable} model training...", 'TRAINING')
    model = _create_model(training_config, vocab_sizes, output_variable)
    optimizer, loss_fn = create_optimizer_and_loss(model, training_config)
    trainer = ModelTrainer(
        model, optimizer, loss_fn, training_config, run_path,
        output_variable=output_variable, logger=logger, bin_means=bin_means,
    )
    metrics = trainer.train_with_shards(train_shards_dir=train_shards_dir)
    del trainer, model, optimizer, loss_fn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ===========================================================================
# Stage: training
# ===========================================================================
def stage_training(config, run_path):
    """Train separate electricity and gas GPT models."""
    training_config = config['training']
    logger = setup_json_logging("experiment_log", run_path)
    save_experiment_config(config, run_path)
    set_seed(training_config['random_seed'])

    # Load tokenisers and compute vocab sizes
    log("Loading tokenisers...", 'TRAINING')
    tok = _load_tokenizers_and_vocab(training_config['data']['tokeniser_file'])

    train_shards_dir = training_config['data']['path_to_shard_data']

    # Load conditioning-variable tokenisers produced during sharding
    with open(os.path.join(train_shards_dir, 'conditioning_variable_tokenisers.pkl'), 'rb') as f:
        conditioning_tokenisers = pickle.load(f)

    # Build vocab_sizes: 7 temporal features + conditioning variables
    vocab_sizes = [
        len(tok['elec_boundaries']) + 3,   # 0: electricity
        len(tok['gas_boundaries']) + 4,    # 1: gas (zero-aware scheme)
        len(tok['temp_boundaries']) + 3,   # 2: temperature
        len(tok['solar_boundaries']) + 3,  # 3: solar
        48 + 1,                            # 4: half-hour of day
        7 + 1,                             # 5: day of week
        12 + 1,                            # 6: month
    ]

    # Conditioning variables come from shard metadata (region/has_pv excluded)
    shard_metadata_path = os.path.join(train_shards_dir, 'shard_metadata.json')
    if os.path.exists(shard_metadata_path):
        with open(shard_metadata_path, 'r') as f:
            shard_metadata = json.load(f)
        conditioning_vars = shard_metadata.get('sharding_config', {}).get(
            'data_filtering', {}).get('conditioning_variables', [])
    else:
        conditioning_vars = config.get('sharding', {}).get(
            'data_filtering', {}).get('conditioning_variables', [])
    training_vars = [v for v in conditioning_vars if v not in ['region', 'has_pv']]

    for var in training_vars:
        if var not in conditioning_tokenisers:
            raise ValueError(f"Conditioning variable '{var}' not found in tokenisers")
        vocab_sizes.append(conditioning_tokenisers[var]['vocab_size'])
    log(f"Total input features: {len(vocab_sizes)}", 'TRAINING')

    elec_metrics = _train_one('electricity', training_config, run_path, vocab_sizes,
                              tok['elec_bin_means'], train_shards_dir, logger)
    gas_metrics = _train_one('gas', training_config, run_path, vocab_sizes,
                             tok['gas_bin_means'], train_shards_dir, logger)

    log(f"Electricity best val loss: {elec_metrics['best_val_loss']:.4f}", 'RESULTS')
    log(f"Gas best val loss: {gas_metrics['best_val_loss']:.4f}", 'RESULTS')
    log("Training completed successfully", 'TRAINING')
    return True


# ===========================================================================
# Stage: inference
# ===========================================================================
def stage_inference(config, run_path):
    """Generate synthetic data with the trained models (unmasked mode)."""
    inf_config = config.get('inference', {}).copy()
    logger = setup_json_logging("inference_log", run_path)
    set_seed(inf_config.get('random_seed', 42))

    for key in ['path_to_shard_data', 'path_to_trained_models', 'path_to_tokeniser_file']:
        if key in inf_config:
            inf_config[key] = resolve(run_path, inf_config[key])
    if 'data' in inf_config and 'path_to_module_1_data' in inf_config['data']:
        inf_config['data']['path_to_module_1_data'] = resolve(
            run_path, inf_config['data']['path_to_module_1_data'])
    inf_config['generated_data_path'] = resolve(
        run_path, inf_config.get('generated_data_path', 'generated_data'))

    shards_dir = inf_config['path_to_shard_data']
    model_dir = inf_config['path_to_trained_models']
    model_type = inf_config.get('best_model_to_use_for_eval', 'val')
    device = get_device_from_config(inf_config, verbose=True)

    elec_path = os.path.join(model_dir, f'electricity_best_{model_type}_model.pth')
    gas_path = os.path.join(model_dir, f'gas_best_{model_type}_model.pth')
    if not (os.path.exists(elec_path) and os.path.exists(gas_path)):
        log(f"Trained models not found in {model_dir}", 'INFERENCE')
        return False

    log("Loading trained models...", 'INFERENCE')
    electricity_model = torch.load(elec_path, map_location=device, weights_only=False)
    gas_model = torch.load(gas_path, map_location=device, weights_only=False)
    electricity_model.eval()
    gas_model.eval()

    tok = load_tokenisers(inf_config['path_to_tokeniser_file'])
    (elec_boundaries, gas_boundaries, temp_boundaries, solar_boundaries,
     elec_bin_means, gas_bin_means, value_transform) = tok

    generator = SyntheticDataGenerator(
        electricity_model=electricity_model,
        gas_model=gas_model,
        elec_boundaries=elec_boundaries,
        gas_boundaries=gas_boundaries,
        temp_boundaries=temp_boundaries,
        solar_boundaries=solar_boundaries,
        elec_bin_means=elec_bin_means,
        gas_bin_means=gas_bin_means,
        config=inf_config,
        logger=logger,
        value_transform=value_transform,
    )

    # Conditioning-variable order (from training config, then shard metadata)
    conditioning_var_order = None
    training_config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        conditioning_var_order = training_config.get('sharding', {}).get(
            'data_filtering', {}).get('conditioning_variables', [])
    if not conditioning_var_order:
        shard_metadata_path = os.path.join(shards_dir, 'shard_metadata.json')
        with open(shard_metadata_path, 'r') as f:
            shard_metadata = json.load(f)
        conditioning_var_order = shard_metadata.get('sharding_config', {}).get(
            'data_filtering', {}).get('conditioning_variables', [])

    with open(os.path.join(shards_dir, 'conditioning_variable_tokenisers.pkl'), 'rb') as f:
        conditioning_variable_tokenisers = pickle.load(f)

    tokenised_data = inf_config.get('data', {}).get('path_to_module_1_data')
    if not tokenised_data:
        tokenised_data = os.path.join(os.path.dirname(shards_dir), 'tokenised_data')

    log("Generating synthetic data (unmasked)...", 'INFERENCE')
    result = generator.generate_sequences_unmasked(
        path_to_shard_data=shards_dir,
        path_to_tokenised_original_data=tokenised_data,
        conditioning_variable_tokenisers=conditioning_variable_tokenisers,
        conditioning_var_order=conditioning_var_order,
    )
    logger.log_stage_end('inference')
    log(f"Generated {result['num_generated']}/{result['num_requested']} households", 'RESULTS')
    log(f"Results saved to: {inf_config['generated_data_path']}", 'RESULTS')
    return True


# ===========================================================================
# Stage: evaluation
# ===========================================================================
def stage_evaluation(config, run_path):
    """Evaluate the fidelity of the generated synthetic data."""
    eval_config = config.get('evaluation', {}).copy()
    set_seed(eval_config.get('random_seed', 42))

    for key in ['generated_data_folder', 'original_data_folder', 'shard_metadata_path',
                'metadata_catalog_path', 'tokeniser_file']:
        if eval_config.get(key) is not None:
            eval_config[key] = resolve(run_path, eval_config[key])

    eval_logger = setup_json_logging("evaluation_log", run_path)
    eval_logger.log_stage_start('evaluation')

    generated = eval_config.get('generated_data_folder')
    original = eval_config.get('original_data_folder')
    shard_metadata_path = eval_config.get('shard_metadata_path')

    # Train/val/test PUPRN splits from shard metadata
    train_puprns = val_puprns = test_puprns = None
    if shard_metadata_path and os.path.exists(shard_metadata_path):
        with open(shard_metadata_path, 'r') as f:
            shard_metadata = json.load(f)
        train_puprns = []
        for shard in shard_metadata.get('training_shards', []):
            train_puprns.extend(shard.get('puprn_list', []))
        for split, target in [('validation_shards', 'val'), ('test_shards', 'test')]:
            shards = shard_metadata.get(split, {})
            puprns = None
            if 'unmasked' in shards:
                puprns = shards['unmasked'].get('puprn_list', [])
            elif shards:
                puprns = shards[list(shards.keys())[0]].get('puprn_list', [])
            if target == 'val':
                val_puprns = puprns
            else:
                test_puprns = puprns

    sampling = eval_config.get('sampling', {})
    mr_config = {
        'acf_lags': eval_config.get('acf_lags', [1, 2, 48, 96, 336]),
        'conditioning_vars': eval_config.get('conditioning_vars', []),
        'real_data_batch_size': eval_config.get('real_data_batch_size', 100),
        'summary_statistics': eval_config.get('summary_statistics', {}),
        'distance_analysis': eval_config.get('distance_analysis', {}),
        'diagnostics_integration': eval_config.get('diagnostics_integration', {}),
        'tokeniser_file': eval_config.get('tokeniser_file'),
    }

    evaluator = ParallelEvaluator(config=mr_config, n_workers=eval_config.get('n_workers'))
    real_dir = original if (original and os.path.exists(original)) else None
    eval_output_dir = os.path.join(run_path, 'evaluation')

    results = evaluator.evaluate(
        synthetic_dir=generated,
        real_train_dir=real_dir,
        real_test_dir=real_dir,
        output_dir=eval_output_dir,
        run_visualizations=eval_config.get('run_visualizations', True),
        train_puprns=train_puprns,
        val_puprns=val_puprns,
        test_puprns=test_puprns,
        train_sample=sampling.get('train_households'),
        val_sample=sampling.get('val_households'),
        test_sample=sampling.get('test_households'),
        synth_batch_limit=sampling.get('synth_batches'),
        metadata_catalog_path=eval_config.get('metadata_catalog_path'),
    )

    if eval_config.get('save_results_json', True):
        output_path = Path(eval_output_dir) / 'parallel_evaluation_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        log(f"Evaluation results saved to: {output_path}", 'EVALUATION')

    # Generate the fidelity-report visualisations from the saved results
    if eval_config.get('run_visualizations', True) and eval_config.get('save_results_json', True):
        results_path = os.path.join(eval_output_dir, 'parallel_evaluation_results.json')
        viz_dir = os.path.join(eval_output_dir, 'visualizations')
        if os.path.exists(results_path):
            try:
                visualize_results(results_path, viz_dir)
                log(f"Visualisations saved to: {viz_dir}", 'EVALUATION')
            except Exception as e:
                log(f"Results visualisation failed: {e}", 'EVALUATION')

    eval_logger.log_stage_end('evaluation')
    log(f"Evaluation complete: {results.n_batches} batches, "
        f"{results.n_households} households", 'EVALUATION')
    return True


STAGE_FUNCS = {
    'dummy_data': stage_dummy_data,
    'tokeniser': stage_tokeniser,
    'sharding': stage_sharding,
    'training': stage_training,
    'inference': stage_inference,
    'evaluation': stage_evaluation,
}


def main():
    parser = argparse.ArgumentParser(description="Run the AI-EDOL pipeline")
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML')
    parser.add_argument('--stages', nargs='*', choices=VALID_STAGES,
                        help='Subset of stages to run (default: experiment.stages)')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    experiment = config.get('experiment', {})
    stages = args.stages or experiment.get('stages', [])
    if not stages:
        print("[ERROR] No stages specified in config or on the command line")
        sys.exit(1)

    run_path = os.path.abspath(experiment.get('output_dir', 'output'))
    os.makedirs(run_path, exist_ok=True)
    log(f"Run directory: {run_path}")
    log(f"Stages: {', '.join(stages)}")

    # Resolve training data paths relative to the run directory
    training_data = config.get('training', {}).get('data', {})
    for key in list(training_data.keys()):
        if key.startswith('path_to') or key == 'tokeniser_file':
            training_data[key] = resolve(run_path, training_data[key])
    config.setdefault('training', {})['run_path'] = run_path

    save_experiment_config(config, run_path)

    for stage in stages:
        log(f"=== Stage: {stage} ===")
        if not STAGE_FUNCS[stage](config, run_path):
            print(f"[PIPELINE] Failed at stage: {stage}")
            sys.exit(1)

    log("Pipeline completed successfully")


if __name__ == '__main__':
    main()
