from smart_open import open
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw', help="Path to input folder", required=True)
parser.add_argument('--outto', '-o', help="Where to store outputs?", required=True)
parser.add_argument('--model', type=str, default='gsd', choices=['ancora', 'gsd'])
parser.add_argument('--lempos', action="store_true", help="Boolean flag")
parser.add_argument('--thres', type=float, default=0.5)
parser.add_argument('--langs', nargs='+', default=['ru', 'en'], help='Example: --langs ru en')

args = parser.parse_args()
lang = args.lang
print(langs)
print(args.lempos)
print(args.thres)
print(args.thres)