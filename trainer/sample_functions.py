"""A simple main file to showcase the template."""

import logging.config
import argparse
import logging.config

"""
This module is an example for a single Python application with some
top level functions. The tests directory includes some unitary tests
for these functions.

This is one of two main files samples included in this
template. Please feel free to remove this, or the other
(sklearn_main.py), or adapt as you need.
"""

def train_and_evaluate(batch_size, epochs, job_dir, output_path):
    pass

def main():
    """Entry point for your module"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type= int, help='Batch size for training')
    parser.add_argument('--epochs', type= int, help='Epochs size for training')
    parser.add_argument('--batch-size', type= int, help='Batch size for training')
    parser.add_argument('--job-dir', default = None, required=False, help = 'Option for GCP')
    parser.add_argument('--model-output-path', help = 'Path to write the SaveModel format')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path)

if __name__ == "__main__":
    main()





  
   