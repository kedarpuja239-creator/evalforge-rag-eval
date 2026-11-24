"""promptfoo exec target: reads the attack prompt from argv[1], prints the assistant's answer."""
import sys

from .rag import ask

if __name__ == "__main__":
    print(ask(sys.argv[1])[0])
