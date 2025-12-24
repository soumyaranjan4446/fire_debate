from fire_debate.scripts.predict_claim import predict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("claim", nargs="?", type=str, help="Claim text")
    parser.add_argument("--true", action="store_true", help="Assert ground truth is True")
    parser.add_argument("--false", action="store_true", help="Assert ground truth is False")
    
    args = parser.parse_args()
    
    if args.claim:
        gt = True if args.true else (False if args.false else None)
        predict(args.claim, gt)
    else:
        # Default test if no arg provided
        predict("Camels don't need food for survival", False)