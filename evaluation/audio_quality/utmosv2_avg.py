import os
import argparse
import utmosv2


def compute_average_mos(wav_dir):
    # Load the pretrained UT-MOS model once
    model = utmosv2.create_model(pretrained=True)

    mos_scores = []
    # Walk through the directory and find all .wav files
    for root, _, files in os.walk(wav_dir):
        for fname in files:
            if fname.lower().endswith('.wav'):
                path = os.path.join(root, fname)
                try:
                    mos = model.predict(input_path=path)
                    print(f"{path}: MOS = {mos:.3f}")
                    mos_scores.append(mos)
                except Exception as e:
                    print(f"[ERROR] Could not process {path}: {e}")

    if not mos_scores:
        print("No WAV files found.")
        return

    avg_mos = sum(mos_scores) / len(mos_scores)
    print(f"\nProcessed {len(mos_scores)} files.")
    print(f"Average MOS: {avg_mos:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute average MOS over all WAV files in a directory"
    )
    p.add_argument(
        "--wav-dir", "-d",
        required=True,
        help="Path to the directory containing WAV files"
    )
    args = p.parse_args()
    compute_average_mos(args.wav_dir)
