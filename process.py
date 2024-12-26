import segmentation as segment
import eval as eval

def process():
    images = [
        "8-28-23-HerbsMaize-InHouse_828-091-BeanDragon-WW_2023-08-29_13-35-50.743_17880100_Vis_SV_0",
        "8-28-23-HerbsMaize-InHouse_828-091-BeanDragon-WW_2023-09-09_11-48-42.967_18030200_Vis_SV_0",
        "8-28-23-HerbsMaize-InHouse_828-096-BeanDragon-D_2023-09-07_12-02-55.007_17996600_Vis_SV_0",
        "8-28-23-HerbsMaize-InHouse_828-100-BeanDragon-D_2023-09-04_12-14-10.247_17961000_Vis_SV_0",
        "8-28-23-HerbsMaize-InHouse_913-142-BushCuc-WW_2023-09-28_13-39-39.858_18335300_Vis_SV_0",
        "8-28-23-HerbsMaize-InHouse_913-160-BushCuc-D_2023-10-26_10-30-38.555_18969600_Vis_SV_0"
    ]
    
    segment.main(images)
    
    eval.main(images)
    
process()