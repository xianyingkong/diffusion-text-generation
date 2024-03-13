from model_arch.tokenizer import create_tokenizer
import argparse


parser = argparse.ArgumentParser(
            prog="Tokenizer Creator",
            description="Creates a tokenizer for your corpus file")

parser.add_argument('filepath', type=str)
parser.add_argument('-vs', '--vocab_size', default=30000, type=int)
parser.add_argument('-s', '--save_path', default='tokenizer/custom', type=str)
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    # Get args
    fp = args.filepath
    vs = args.vocab_size
    sp = args.save_path

    print(f"Generating corpus for file {args.filepath}")

    # Call the create token function
    create_tokenizer(fp, vs, save_folder=sp)

    print('Tokenizer created')
    
