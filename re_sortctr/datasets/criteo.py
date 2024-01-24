


import numpy as np
from re_sortctr.preprocess import FeatureProcessor as BaseFeatureProcessor


class FeatureProcessor(BaseFeatureProcessor):
    def convert_to_bucket(self, df, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return df[col_name].map(_convert_to_bucket).astype(int)


	


