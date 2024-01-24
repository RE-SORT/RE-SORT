
import pandas as pd
from re_sortctr.preprocess import FeatureProcessor as BaseFeatureProcessor

class FeatureProcessor(BaseFeatureProcessor):
    def extract_country_code(self, df, col_name):
        return df[col_name].apply(lambda isrc: isrc[0:2] if not pd.isnull(isrc) else "")

    def bucketize_age(self, df, col_name):
        def _bucketize(age):
            if pd.isnull(age):
                return ""
            else:
                age = float(age)
                if age < 1 or age > 95:
                    return ""
                elif age <= 10:
                    return "1"
                elif age <=20:
                    return "2"
                elif age <=30:
                    return "3"
                elif age <=40:
                    return "4"
                elif age <=50:
                    return "5"
                elif age <=60:
                    return "6"
                else:
                    return "7"
        return df[col_name].apply(_bucketize)


