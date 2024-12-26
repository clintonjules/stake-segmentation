import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from sklearn.metrics import confusion_matrix
from PIL import Image

class SegmentationEvaluator:
    """
    Evaluate image segmentation results using various metrics and compares predicted segmentation masks against ground truth masks.
    """
    def __init__(self, pred_mask, true_mask):
        self.pred_mask = pred_mask
        self.true_mask = true_mask
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            true_mask.ravel(),
            pred_mask.ravel()
        ).ravel()

    def iou(self):
        """
        Intersection over Union (IoU) / Jaccard index. Returns ratio of intersection to union of predicted and ground truth mask.
        """
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        return intersection / union if union != 0 else 0

    def dice_coefficient(self):
        # Dice coefficient (F1 score). Returns harmonic mean of precision and recall.
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn) if (2 * self.tp + self.fp + self.fn) != 0 else 0

    def precision(self):
        # Positive predictive value
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) != 0 else 0

    def recall(self):
        # Sensitivity
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @staticmethod
    def _get_boundary(mask):
        # Get boundary from mask with morphological operations
        dilated = binary_dilation(mask)
        eroded = binary_erosion(mask)
        return dilated ^ eroded

    def get_all_metrics(self):
            return {
                'IoU': self.iou(),
                'Dice': self.dice_coefficient(),
                'Recall': self.recall(),
                'Precision': self.precision(),
                'Accuracy': self.accuracy()
            }

    def visualize_results(self, save_path=None):
        # Return visualization of segmentation results.
       fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

       ax1.imshow(self.true_mask, cmap='gray')
       ax1.set_title('Ground Truth')
       ax1.axis('off')

       ax2.imshow(self.pred_mask, cmap='gray')
       ax2.set_title('Prediction')
       ax2.axis('off')

       overlap = np.zeros((*self.true_mask.shape, 3))
       overlap[self.true_mask == 1] = [0, 1, 0]
       overlap[self.pred_mask == 1] = [1, 0, 0]
       overlap[(self.true_mask == 1) & (self.pred_mask == 1)] = [1, 1, 0]

       ax3.imshow(overlap)
       ax3.set_title('Overlay')
       ax3.axis('off')

       from matplotlib.patches import Patch
       legend_elements = [
           Patch(facecolor='yellow', label='True Positive'),
           Patch(facecolor='red', label='False Positive'),
           Patch(facecolor='green', label='False Negative')
       ]

       ax3.legend(handles=legend_elements,
                 loc='center',
                 bbox_to_anchor=(0.5, -0.15),
                 ncol=1)

       plt.tight_layout()

       if save_path:
           plt.savefig(save_path,
                      bbox_inches='tight',
                      dpi=300,
                      pad_inches=0.5)

           overlay_path = save_path.replace('.png', '_overlay.png')
           fig_overlay, ax_overlay = plt.subplots(figsize=(8, 8))
           ax_overlay.imshow(overlap)
           ax_overlay.set_title('Segmentation Overlay')
           ax_overlay.axis('off')
           ax_overlay.legend(handles=legend_elements,
                            loc='center',
                            bbox_to_anchor=(0.5, -0.1),
                            ncol=1)
           plt.tight_layout()
           plt.savefig(overlay_path,
                       bbox_inches='tight',
                       dpi=300,
                       pad_inches=0.5)
           plt.close(fig_overlay)

       plt.close(fig)

def evaluate_segmentation(pred_img, gt_img, output_folder):
    # Evaluate segmentation results for all images in given folder
    os.makedirs(output_folder, exist_ok=True)
    result_file = os.path.join(output_folder, "Segmentation_Evaluation_Results.txt")

    with open(result_file, "w") as f:
            print(f"Processing:\nPrediction: {pred_img}\nGround Truth: {gt_img}")

            pred_img = Image.open(pred_img).convert('L')
            true_img = Image.open(gt_img).convert('L')

            pred_mask = np.array(pred_img) > 127
            true_mask = np.array(true_img) > 127

            evaluator = SegmentationEvaluator(pred_mask, true_mask)
            metrics = evaluator.get_all_metrics()

            f.write(f"File: {pred_img}\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            f.write("\n")

            eval_visualization_path = os.path.join(
                output_folder,
                "combined_mask_evaluation_visualization.png"
            )
            evaluator.visualize_results(save_path=eval_visualization_path)
            
def main(images):
    for image in images:
        prediction = "segmentation_output/" + image + "/final_mask.png"
        ground_truth = "ground_truth/" + image + "_ground_truth.png"
        output_folder = "evaluation/" + image
        
        evaluate_segmentation(prediction, ground_truth, output_folder)