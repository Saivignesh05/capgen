"""
evaluation.py - Stage 9: Evaluation & Human Verification Engine.
Compares automatic event detections against human-verified ground-truth annotations.
Calculates collar-based Precision, Recall, F1 score, Onset MAE, Offset MAE,
and per-class performance tables.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import numpy as np


class EventEvaluator:
    def __init__(self, collar_sec: float = 0.200):
        """
        Args:
            collar_sec: Tolerance window (in seconds) for onset/offset event boundary matching.
        """
        self.collar_sec = collar_sec

    def evaluate_clip(
        self,
        predicted_events: List[Dict[str, Any]],
        ground_truth_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates a single clip against its ground-truth annotations.
        """
        unverified = [
            event for event in ground_truth_events
            if event.get("verification_status") != "human_verified"
        ]
        if unverified:
            raise ValueError(
                "Ground truth contains unverified events; manually verify the "
                "template before scoring."
            )

        classes = sorted(list(set(
            [e["event_label"] for e in predicted_events] +
            [e["event_label"] for e in ground_truth_events]
        )))

        overall_tp = 0
        overall_fp = 0
        overall_fn = 0
        onset_errors = []
        offset_errors = []

        per_class_results = {}

        for c in classes:
            preds_c = [e for e in predicted_events if e["event_label"] == c]
            gts_c = [e for e in ground_truth_events if e["event_label"] == c]

            tp, fp, fn, onsets, offsets = self._match_events(preds_c, gts_c)

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class_results[c] = {
                "ground_truth_count": len(gts_c),
                "predicted_count": len(preds_c),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "mean_onset_error_sec": round(float(np.mean(onsets)), 4) if onsets else None,
                "mean_offset_error_sec": round(float(np.mean(offsets)), 4) if offsets else None
            }

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            onset_errors.extend(onsets)
            offset_errors.extend(offsets)

        total_prec = (overall_tp / (overall_tp + overall_fp)) if (overall_tp + overall_fp) > 0 else 0.0
        total_rec = (overall_tp / (overall_tp + overall_fn)) if (overall_tp + overall_fn) > 0 else 0.0
        total_f1 = (2 * total_prec * total_rec / (total_prec + total_rec)) if (total_prec + total_rec) > 0 else 0.0

        return {
            "evaluation_collar_seconds": self.collar_sec,
            "overall_metrics": {
                "total_ground_truth": len(ground_truth_events),
                "total_predictions": len(predicted_events),
                "true_positives": overall_tp,
                "false_positives": overall_fp,
                "false_negatives": overall_fn,
                "precision": round(total_prec, 4),
                "recall": round(total_rec, 4),
                "f1_score": round(total_f1, 4),
                "mean_onset_error_seconds": round(float(np.mean(onset_errors)), 4) if onset_errors else None,
                "mean_offset_error_seconds": round(float(np.mean(offset_errors)), 4) if offset_errors else None
            },
            "per_class_breakdown": per_class_results
        }

    def _match_events(
        self,
        preds: List[Dict[str, Any]],
        gts: List[Dict[str, Any]]
    ) -> Tuple[int, int, int, List[float], List[float]]:
        """
        Bipartite collar-based matching between predicted intervals and ground truth intervals.
        """
        matched_gt = set()
        matched_pred = set()
        onset_diffs = []
        offset_diffs = []

        # Find valid matches
        for p_idx, p in enumerate(preds):
            best_gt_idx = None
            best_dist = float("inf")

            for g_idx, g in enumerate(gts):
                if g_idx in matched_gt:
                    continue

                # Collar match: onset and offset within tolerance or significant temporal overlap
                onset_err = abs(p["start_time"] - g["start_time"])
                offset_err = abs(p["end_time"] - g["end_time"])

                overlap_start = max(p["start_time"], g["start_time"])
                overlap_end = min(p["end_time"], g["end_time"])
                has_overlap = (overlap_end - overlap_start) > 0

                if (onset_err <= self.collar_sec and offset_err <= self.collar_sec) or has_overlap:
                    total_dist = onset_err + offset_err
                    if total_dist < best_dist:
                        best_dist = total_dist
                        best_gt_idx = g_idx

            if best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                g_match = gts[best_gt_idx]
                onset_diffs.append(abs(p["start_time"] - g_match["start_time"]))
                offset_diffs.append(abs(p["end_time"] - g_match["end_time"]))

        tp = len(matched_pred)
        fp = len(preds) - tp
        fn = len(gts) - len(matched_gt)

        return tp, fp, fn, onset_diffs, offset_diffs

    @staticmethod
    def create_verification_template(predictions_json_path: str | Path, output_gt_path: str | Path):
        """
        Creates an editable ground-truth reference file from predictions
        allowing researchers to verify, correct boundaries, or add missing events.
        """
        with open(predictions_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        review_queue = []
        review_items = list(data.get("timeline", []))
        review_items.extend(data.get("sound_event_candidates", []))
        for item in review_items:
            review_queue.append({
                "event_id": item.get("event_id", f"candidate_{len(review_queue) + 1:03d}"),
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "duration": item.get("duration", item["end_time"] - item["start_time"]),
                "event_category": item.get("event_category", "sound_event"),
                "event_label": item.get("event_label", "unknown"),
                "confidence": item.get("confidence", 0.0),
                "verification_status": "unverified",
                "human_annotator_notes": "Review and verify this candidate before scoring"
            })

        output_payload = {
            "clip_id": data["clip_id"],
            "annotator": "researcher_manual_audit_pending",
            "verification_required": True,
            "ground_truth_events": [],
            "review_queue": review_queue,
            "note": "Move only manually confirmed events from review_queue into ground_truth_events.",
        }

        with open(output_gt_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        return output_gt_path
