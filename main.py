
import configargparse
import logging
import os
import sys
import wandb
import re
from typing import Sequence, Type
import importlib.util
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)

from sky_spot import env as env_lib
from sky_spot.env import MultiTraceEnv, TraceEnv 
from sky_spot import simulate
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.task import SingleTask, Task

# Default to offline: evaluators have no W&B account, and artifacts should
# not send data to external services during review. Set WANDB_MODE=online
# explicitly to sync runs.
os.environ.setdefault('WANDB_MODE', 'offline')
# Allow disabling Weights & Biases in restricted environments
if os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
    try:
        wandb.init(project='sky-spot')
    except Exception:
        # Fall back silently when service cannot be started (e.g., sandboxed runs)
        pass
logger = logging.getLogger(__name__)
    
def load_strategy_from_file(file_path: str) -> Type[strategy_lib.Strategy]:
    """Dynamically loads a strategy class from a Python file."""
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all strategy classes, but exclude base classes
        found_classes = []
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, strategy_lib.Strategy) and obj is not strategy_lib.Strategy:
                # Also exclude MultiRegionStrategy base class
                # And only consider classes DEFINED in this module (ignore imported bases)
                if (
                    obj.__name__ not in ['Strategy', 'MultiRegionStrategy']
                    and getattr(obj, '__module__', None) == module.__name__
                ):
                    found_classes.append(obj)
                    logger.info(f"Found strategy class '{obj.__name__}' in {file_path}")
        
        if found_classes:
            # Return the first non-base strategy class found
            return found_classes[0]

        raise AttributeError(f"No concrete strategy class found in {file_path}")
    except Exception as e:
        logger.error(f"Failed to load strategy from {file_path}: {e}")
        raise


def find_indexed_traces(dir_path):
    indexed_files = {}
    if not os.path.isdir(dir_path):
        logger.warning(f"Path is not a directory: {dir_path}")
        return indexed_files
    try:
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                match = re.match(r"(\d+)\.json", filename)
                if match:
                    index = int(match.group(1))
                    indexed_files[index] = os.path.join(dir_path, filename)
    except OSError as e:
        logger.error(f"Error reading directory {dir_path}: {e}")
    return indexed_files


if __name__ == '__main__':
    root_logger = logging.getLogger('sky_spot')

    def setup_logger():
        logging_level = os.environ.get('LOG_LEVEL', 'DEBUG')
        handler = logging.StreamHandler(sys.stdout)
        
        class SimpleColorFormatter(logging.Formatter):
            def format(self, record):
                # Get the last part of the module name
                name_parts = record.name.split('.')
                short_name = name_parts[-1] if name_parts else record.name
                
                # Apply colors based on level
                if record.levelname == 'ERROR':
                    color = Fore.RED
                elif record.levelname == 'WARNING':
                    color = Fore.YELLOW
                elif record.levelname == 'DEBUG':
                    color = Fore.LIGHTBLACK_EX  # Gray
                else:
                    color = ''  # No color for INFO
                
                return f"{color}[{short_name}] {record.getMessage()}{Style.RESET_ALL}"
        
        handler.setFormatter(SimpleColorFormatter())
        handler.setLevel(logging_level)
        root_logger.setLevel(logging_level)
        root_logger.addHandler(handler)

    setup_logger()

    parser = configargparse.ArgumentParser('Skypilot spot simulator')

    parser.add_argument('--config',
                        type=str,
                        default=None,
                        is_config_file=True,
                        required=False)
    group = parser.add_argument_group('Global options')
    group.add_argument('--deadline-hours',
                       type=float,
                       default=10,
                       help='Deadline of the task in hours')
    group.add_argument(
        '--task-duration-hours',
        type=float,
        nargs='+',
        default=[10],
        help=
        'Duration(s) of task(s) in hours. For chained tasks, provide multiple values.'
    )
    group.add_argument(
        '--restart-overhead-hours',
        type=float,
        nargs='+',
        default=[0.2],
        help=
        'Overhead(s) of restarting tasks in hours. Provide multiple values for different tasks.'
    )
    group.add_argument(
        '--checkpoint-size-gb',
        type=float,
        default=50.0,
        help='Size of the checkpoint in GB for calculating migration times (default: 50GB)'
    )
    group.add_argument('--cross-region-migration-time',
                       action='store_true',
                       help='If set, add extra downtime for cross-region migrations (default behaviour overlaps the transfer).')
    group.add_argument('--output-dir',
                       type=str,
                       default='exp/',
                       help='Output directory')
    group.add_argument('--no-history',
                       action='store_true',
                       help='Do not record per-tick history to drastically reduce memory usage')
    
    # --- MODIFIED PART: Define strategy args at the top level ---
    strategy_group = parser.add_argument_group('Strategy Selection')
    strategy_group.add_argument('--strategy-file',
                       type=str,
                       default=None,
                       help='Path to a Python file defining the strategy to use.')
    strategy_group.add_argument('--strategy',
                            type=str,
                            default='strawman',
                            choices=strategy_lib.Strategy.SUBCLASSES.keys(),
                            help='Name of the built-in strategy to use.')
    # --- END MODIFIED PART ---

    args, _ = parser.parse_known_args()

    # Simplified CLI-only flow: YAML scenarios and sub-task multi-env are not supported.
    envs = env_lib.Env.from_args(parser)

    if args.strategy_file:
        logger.info(f"Dynamically loading strategy from: {args.strategy_file}")
        StrategyClass = load_strategy_from_file(args.strategy_file)
    else:
        logger.info(f"Loading built-in strategy: {args.strategy}")
        StrategyClass = strategy_lib.Strategy.get(args.strategy)

    strategy = StrategyClass._from_args(parser)
    args = strategy.args

    assert envs, "Env.from_args did not return any environments based on CLI arguments."
    # Enforce single-task (no sub_task/chained task support)
    assert len(args.task_duration_hours) == 1, "Chained/sub_task is not supported; provide a single duration."
    current_task = SingleTask(config={
        'duration': args.task_duration_hours[0],
        'checkpoint_size_gb': args.checkpoint_size_gb
    })
    logger.info(f"Task: {current_task}")

    trace_param: str
    if hasattr(args, 'trace_file') and args.trace_file:
        # Use region dir + filename to avoid collisions
        trace_param = f"{os.path.basename(os.path.dirname(args.trace_file))}_{os.path.basename(args.trace_file)}"
        logger.info(f"Using trace_file: {trace_param}")
    elif hasattr(args, 'trace_files') and args.trace_files:
        trace_files_str = ",".join([f"{os.path.basename(os.path.dirname(f))}_{os.path.basename(f)}" for f in args.trace_files])
        trace_param = f"multi_region_{trace_files_str}"
        logger.info(f"Using multiple trace files: {args.trace_files}")
        logger.info(f"Combined name: {trace_param}")
    else:
        trace_param = "unknown_cli_trace"


    final_env_start_hours = getattr(args, 'env_start_hours', 0.0)
    assert 'current_task' in locals(), "current_task was not defined before simulate call."

    # Catch a mismatched --strategy/--env pair here so the CLI prints the hint
    # rather than a traceback out of the simulation loop.
    try:
        simulate._check_env_strategy_compatibility(envs, strategy)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Starting simulation with {len(envs)} environment instance(s)...")
    simulate.simulate(envs=envs,
                      strategy=strategy,
                      task=current_task,
                      trace_file=trace_param,
                      deadline_hours=strategy.args.deadline_hours,
                      restart_overhead_hours=strategy.args.restart_overhead_hours,
                      env_start_hours=final_env_start_hours,
                      output_dir=args.output_dir,
                      kwargs=vars(strategy.args),
                      dump_history=(not getattr(args, 'no_history', False)))
