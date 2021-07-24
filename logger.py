import logging
import os
from pathlib import Path
import json
from datetime import datetime
from params import args

# set logger
dt = datetime.now()
dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
logger = logging.getLogger('__main__')  # this is the global logger

# set output directory
output_dir = os.path.join('outputs', args.exp_name)
Path(output_dir).mkdir(parents=True, exist_ok=True)
setattr(args, 'output_dir', output_dir)

# automatically resume from the last checkpoint
checkpoint_path = os.path.join(output_dir, 'last_checkpoint.pth')
if os.path.exists(checkpoint_path) and not args.resume:
    setattr(args, 'resume', checkpoint_path)

if not args.eval:
    log_path = os.path.join(output_dir, 'all_logs.txt')
    with open(os.path.join(args.output_dir, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, indent=4)
else:
    log_path = os.path.join(output_dir, 'eval_logs.txt')

fh = logging.FileHandler(log_path, 'a+')
formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
fh.setFormatter(formatter)
logger.addHandler(fh)