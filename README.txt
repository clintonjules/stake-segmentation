Data:
	plant_images - Contains the images used for segmentation
	ground_truth - Contains the ground truth segmentation area for each of the plant images used to score the accuracy of the operation

Code:
	segmentation.py - Performs the segmentation process for the images
	The segment function does the operations and takes an image file path as input
	The main function applies the operations to all of the images in the list "images" as input

	eval.py - Performs the evaluation of the segmentation process on the outputs of the segmentation process
	The class SegmentationEvaluator hosts the operations to perform the evaluation calculations
	The function evaluate_segmentation performs the evaluations using the segmented image, found truth image, and an output folder
	The main function applies the operations to all of the images in the list "images" as input

	process.py - Runs both segmentation.py and eval.py all in one file using the plant images

Output:
	segmentation_output - Contains the results of the segmentation process separated by each plant image. Each folder division contains:
		the original image
		the brown color mask
		edge mask
		rectangle crop mask
		morphological operation mask
		segmented image mask
		the combination of each mask
		svg figure showing the six separate masks before combination

	evaluation - Contains the results of the segmentation process separated by each plant image. Each folder division contains:
	
		the final combined mask overlay visualization of the true positive, false positive, and false negative pixels for the final segmentation
		the final combined mask overlay visualization of the true positive, false positive, and false negative pixels for the final segmentation along with the ground truth and final combined mask
		the numerical results of the evaluation including:
			
			IoU
			Dice
			Recall
			Precision
			Accuracy