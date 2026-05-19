"""
JSON-structured logging utilities for TRE compatibility.

Provides structured logging with JSON output format suitable for
Trusted Research Environments and experiment tracking.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import traceback
import numpy as np


def json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = traceback.format_exception(*record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry, default=json_serializer)


class ExperimentLogger:
    """
    Enhanced logger for ML experiment tracking with JSON structure.
    
    Provides experiment lifecycle tracking, metrics logging, and
    TRE-compatible structured output.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 run_path: str):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            run_path: Path to save logs and artifacts
            config: Experiment configuration
        """
        self.experiment_name = experiment_name
        self.run_path = Path(run_path)
        self.start_time = time.time()
        self.experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory
        self.run_path.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        # Log experiment start
        self.log_experiment_start()
    
    def _setup_loggers(self):
        """Setup structured and standard loggers."""
        # JSON structured logger
        self.json_logger = logging.getLogger(f"{self.experiment_name}_json")
        self.json_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.json_logger.handlers.clear()
        
        # JSON file handler
        json_handler = logging.FileHandler(
            self.run_path / 'experiment_log.jsonl', 
            mode='w'
        )
        json_handler.setFormatter(JSONFormatter())
        self.json_logger.addHandler(json_handler)
        
        # Standard text logger (for backward compatibility)
        self.text_logger = logging.getLogger(f"{self.experiment_name}_text")
        self.text_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.text_logger.handlers.clear()
        
        # Text file handler
        text_handler = logging.FileHandler(
            self.run_path / 'logs.txt',
            mode='w'
        )
        text_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.text_logger.addHandler(text_handler)
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.text_logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplicate logs
        self.json_logger.propagate = False
        self.text_logger.propagate = False
    
    def log_experiment_start(self):
        """Log experiment initialization."""
        start_info = {
            'event_type': 'experiment_start',
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'start_timestamp': datetime.fromtimestamp(self.start_time).isoformat(),
            'environment': {
                'python_version': None,  # Can be added if needed
                'torch_version': None,   # Can be added if needed
                'run_path': str(self.run_path)
            }
        }
        
        self._log_structured('info', 'Experiment started', start_info)
    
    def log_stage_start(self, stage_name: str, stage_config: Optional[Dict[str, Any]] = None):
        """Log the start of an experiment stage."""
        stage_info = {
            'event_type': 'stage_start',
            'stage_name': stage_name,
            'stage_timestamp': datetime.now().isoformat(),
            'stage_config': stage_config or {}
        }
        
        self._log_structured('info', f'Stage started: {stage_name}', stage_info)
    
    def log_stage_end(self, stage_name: str, stage_results: Optional[Dict[str, Any]] = None):
        """Log the completion of an experiment stage."""
        stage_info = {
            'event_type': 'stage_end',
            'stage_name': stage_name,
            'stage_timestamp': datetime.now().isoformat(),
            'stage_results': stage_results or {}
        }
        
        self._log_structured('info', f'Stage completed: {stage_name}', stage_info)
    
    def log_data_info(self, 
                      num_households: int,
                      data_shapes: Dict[str, Any],
                      puprn_list: Optional[list] = None):
        """Log data loading and preprocessing information."""
        data_info = {
            'event_type': 'data_info',
            'num_households': num_households,
            'data_shapes': data_shapes,
            'puprn_count': len(puprn_list) if puprn_list else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', 'Data information logged', data_info)
    
    def log_model_info(self, 
                       model_params: int,
                       model_config: Dict[str, Any],
                       vocab_sizes: Optional[list] = None):
        """Log model architecture information."""
        model_info = {
            'event_type': 'model_info',
            'model_parameters': model_params,
            'model_config': model_config,
            'vocab_sizes': vocab_sizes,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', f'Model initialized with {model_params/1e6:.1f}M parameters', model_info)
    
    def log_training_epoch(self, 
                          epoch: int,
                          train_loss: float,
                          val_loss: Optional[float] = None,
                          learning_rate: Optional[float] = None,
                          epoch_time: Optional[float] = None):
        """Log training epoch information."""
        epoch_info = {
            'event_type': 'training_epoch',
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time_seconds': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        
        message = f'Epoch {epoch}: train_loss={train_loss:.4f}'
        if val_loss is not None:
            message += f', val_loss={val_loss:.4f}'
        
        self._log_structured('info', message, epoch_info)
    
    def log_training_summary(self,
                           best_train_loss: float,
                           best_val_loss: float,
                           best_epoch: int,
                           total_training_time: float,
                           early_stopping: bool = False,
                           output_variable: Optional[str] = None):
        """
        Log training completion summary for a model.

        Args:
            best_train_loss: Best training loss achieved.
            best_val_loss: Best validation loss achieved.
            best_epoch: Epoch at which best validation loss was achieved.
            total_training_time: Total training time in seconds.
            early_stopping: Whether early stopping was triggered.
            output_variable: The output variable this model was trained for
                ('electricity' or 'gas'). If provided, included in the log.
        """
        training_summary = {
            'event_type': 'training_complete',
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_training_time_seconds': total_training_time,
            'early_stopping_triggered': early_stopping,
            'timestamp': datetime.now().isoformat()
        }

        # Add output variable if provided (for dual-model training)
        if output_variable is not None:
            training_summary['output_variable'] = output_variable

        message = f'Training completed for {output_variable}' if output_variable else 'Training completed'
        self._log_structured('info', message, training_summary)
    
    def log_generation_info(self, 
                           num_sequences: int,
                           generation_config: Dict[str, Any],
                           generation_time: Optional[float] = None):
        """Log synthetic data generation information."""
        generation_info = {
            'event_type': 'data_generation',
            'num_sequences_generated': num_sequences,
            'generation_config': generation_config,
            'generation_time_seconds': generation_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', f'Generated {num_sequences} synthetic sequences', generation_info)
    
    def log_evaluation_results(self, evaluation_metrics: Dict[str, Any]):
        """Log model evaluation results."""
        eval_info = {
            'event_type': 'evaluation_complete',
            'evaluation_metrics': evaluation_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', 'Model evaluation completed', eval_info)
    
    def log_error(self, error_message: str, error_context: Optional[Dict[str, Any]] = None):
        """Log error information."""
        error_info = {
            'event_type': 'error',
            'error_message': error_message,
            'error_context': error_context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('error', error_message, error_info)
    
    def log_grid_search_start(self, 
                             grid_id: str,
                             parameter_combinations: int,
                             grid_parameters: Dict[str, list]):
        """Log grid search initialization."""
        grid_info = {
            'event_type': 'grid_search_start',
            'grid_id': grid_id,
            'parameter_combinations': parameter_combinations,
            'grid_parameters': grid_parameters,
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', f'Grid search started: {parameter_combinations} combinations', grid_info)
    
    def log_grid_search_run_start(self, 
                                 grid_id: str,
                                 run_index: int,
                                 total_runs: int,
                                 parameter_combination: Dict[str, Any]):
        """Log individual grid search run start."""
        run_info = {
            'event_type': 'grid_search_run_start',
            'grid_id': grid_id,
            'run_index': run_index,
            'total_runs': total_runs,
            'parameter_combination': parameter_combination,
            'random_seed_reset': True,  # Track that seed was reset for this run
            'timestamp': datetime.now().isoformat()
        }
        
        params_str = ', '.join([f'{k}={v}' for k, v in parameter_combination.items()])
        self._log_structured('info', f'Grid run {run_index+1}/{total_runs}: {params_str}', run_info)
    
    def log_grid_search_run_end(self, 
                               grid_id: str,
                               run_index: int,
                               run_results: Dict[str, Any]):
        """Log individual grid search run completion."""
        run_info = {
            'event_type': 'grid_search_run_end',
            'grid_id': grid_id,
            'run_index': run_index,
            'run_results': run_results,
            'timestamp': datetime.now().isoformat()
        }
        
        val_loss = run_results.get('best_val_loss', 'N/A')
        self._log_structured('info', f'Grid run {run_index+1} completed: val_loss={val_loss}', run_info)
    
    def log_grid_search_end(self, 
                           grid_id: str,
                           success: bool = True,
                           best_combination: Optional[Dict[str, Any]] = None,
                           grid_summary: Optional[Dict[str, Any]] = None):
        """Log grid search completion."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        grid_info = {
            'event_type': 'grid_search_end',
            'grid_id': grid_id,
            'success': success,
            'total_grid_time_seconds': total_time,
            'end_timestamp': datetime.fromtimestamp(end_time).isoformat(),
            'best_combination': best_combination or {},
            'grid_summary': grid_summary or {}
        }
        
        status = "successfully" if success else "with errors"
        self._log_structured('info', f'Grid search completed {status}', grid_info)

    def log_experiment_end(self, 
                          success: bool = True,
                          final_results: Optional[Dict[str, Any]] = None):
        """Log experiment completion."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        end_info = {
            'event_type': 'experiment_end',
            'experiment_id': self.experiment_id,
            'success': success,
            'total_experiment_time_seconds': total_time,
            'end_timestamp': datetime.fromtimestamp(end_time).isoformat(),
            'final_results': final_results or {}
        }
        
        status = "successfully" if success else "with errors"
        self._log_structured('info', f'Experiment completed {status}', end_info)
    
    def log_custom_metric(self, 
                         metric_name: str,
                         metric_value: Union[int, float, str],
                         metric_context: Optional[Dict[str, Any]] = None):
        """Log custom metrics."""
        metric_info = {
            'event_type': 'custom_metric',
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_context': metric_context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self._log_structured('info', f'Metric {metric_name}: {metric_value}', metric_info)
    
    def _log_structured(self, level: str, message: str, extra_fields: Dict[str, Any]):
        """Internal method to log with structured data."""
        # Create a custom LogRecord with extra fields
        record = logging.LogRecord(
            name=self.json_logger.name,
            level=getattr(logging, level.upper()),
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        record.extra_fields = extra_fields
        
        # Log to JSON logger
        self.json_logger.handle(record)
        
        # Also log to text logger for backward compatibility
        getattr(self.text_logger, level.lower())(message)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            self._log_structured('info', message, kwargs)
        else:
            self.text_logger.info(message)
            self.json_logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            self._log_structured('warning', message, kwargs)
        else:
            self.text_logger.warning(message)
            self.json_logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            self._log_structured('error', message, kwargs)
        else:
            self.text_logger.error(message)
            self.json_logger.error(message)


def setup_json_logging(experiment_name: str,
                      run_path: str) -> ExperimentLogger:
    """
    Convenience function to setup JSON logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        run_path: Path to save logs
        config: Experiment configuration
        
    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, run_path)