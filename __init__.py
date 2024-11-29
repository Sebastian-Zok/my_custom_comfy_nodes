

from .FeedFlick import LeftImageSelector, RightImageSelector, FillWithColor, DatasetCreator, Number_Counter

NODE_CLASS_MAPPINGS = {
    "Left Image Selector" : LeftImageSelector,
    "Right Image Selector" : RightImageSelector,
    "BG Color Setter": FillWithColor,
    "Dataset Creator": DatasetCreator,
    "NumberCounter": Number_Counter
    
}

__all__ = ['NODE_CLASS_MAPPINGS']

 