import argparse
import io
import json
import sys

import numpy as np
import zmq
from PIL import Image

from problem_reduction.masking.groundedsam import GroundedSam2Tracker


def main():
    parser = argparse.ArgumentParser(description="GroundedSam2Tracker ZMQ server")
    parser.add_argument("--bind", default="tcp://*:5555", help="ZMQ bind endpoint")
    parser.add_argument("--dino_id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    args = parser.parse_args()

    tracker = GroundedSam2Tracker(
        dino_id=args.dino_id,
        sam2_id=args.sam2_id,
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[GroundedSam2TrackerServer] Listening on {args.bind}")
    sys.stdout.flush()

    while True:
        try:
            parts = sock.recv_multipart()
            if not parts:
                sock.send_json({"status": "error", "error": "Empty request"})
                continue

            header = json.loads(parts[0].decode("utf-8"))
            cmd = header.get("cmd")

            if cmd == "ping":
                sock.send_json({"status": "ok"})
                continue

            if cmd == "shutdown":
                sock.send_json({"status": "ok"})
                break

            if cmd == "reset":
                if len(parts) != 2:
                    sock.send_json({"status": "error", "error": "reset expects 2 frames (header, image)"})
                    continue
                text = header.get("text", "")
                frame = _decode_png_to_image(parts[1])
                try:
                    tracker.reset(frame, text)
                except Exception as e:
                    sock.send_json({"status": "error", "error": str(e)})
                    continue
                sock.send_json({"status": "ok"})
                continue

            if cmd == "step":
                if len(parts) != 2:
                    sock.send_json({"status": "error", "error": "step expects 2 frames (header, image)"})
                    continue
                frame = _decode_png_to_image(parts[1])
                try:
                    idx, masks_t = tracker.step(frame)
                    masks_np = masks_t.detach().cpu().numpy().astype(np.bool_)
                    masks_bytes = _save_npy_to_bytes(masks_np)
                    reply_header = json.dumps({"status": "ok", "idx": int(idx)}).encode("utf-8")
                    sock.send_multipart([reply_header, masks_bytes])
                except Exception as e:
                    sock.send_json({"status": "error", "error": str(e)})
                continue

            sock.send_json({"status": "error", "error": f"Unknown cmd: {cmd}"})

        except KeyboardInterrupt:
            sock.send_json({"status": "ok"})
            break
        except Exception as e:
            try:
                sock.send_json({"status": "error", "error": str(e)})
            except Exception:
                pass

    sock.close(linger=0)


def _decode_png_to_image(buf_bytes: bytes) -> Image.Image:
    with io.BytesIO(buf_bytes) as bio:
        img = Image.open(bio)
        img = img.convert("RGB")
    return img


def _save_npy_to_bytes(arr: np.ndarray) -> bytes:
    with io.BytesIO() as bio:
        np.save(bio, arr, allow_pickle=False)
        return bio.getvalue()


if __name__ == "__main__":
    main()


