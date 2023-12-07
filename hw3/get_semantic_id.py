import pandas as pd

def get_semantic_id(target_object):
    df = pd.read_excel('apartment0_classes.xlsx')
    # get the label of target object
    target_label = df[df['Name'] == target_object]['label'].values[0]
    return target_label
    
if __name__ == "__main__":
    get_semantic_id('ceiling')
    