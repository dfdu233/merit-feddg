"""Entry point for the frozen A -> B/C -> independent confirmation chain."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from merit_feddg.algorithm_chain.cli import main
if __name__ == '__main__':
    main()
