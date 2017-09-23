import pandas as pd
from util.normalize import make_normalized_array
from similarity_tests.similarity_metrics import compute_similarity


def perform_similarity(normed_depth, test_surfaces, ids):

    rows_list = []

    for surfs in zip(test_surfaces, ids):
        print("Processing... " + surfs[1])
        results = compute_similarity(normed_depth, make_normalized_array(surfs[0]), surfs[1])
        rows_list.append(results)

    df = pd.DataFrame(rows_list)
    df.set_index('id', inplace=True)
    df['MSE Rank'] = df['mse_value'].rank(ascending=True)
    df['SSIM Rank'] = df['ssim_value'].rank(ascending=False)
    df['CW-SSIM Rank'] = df['cw_ssim_value'].rank(ascending=False)
    df['GMSD Rank'] = df['gms_value'].rank(ascending=True)
    df['FSIM Rank'] = df['fsim_value'].rank(ascending=False)
    df['Avg. Rank'] = (df['MSE Rank'] + df['SSIM Rank'] + df['CW-SSIM Rank'] + df['GMSD Rank'] + df['FSIM Rank']) / 5
    df = df.sort_values(['Avg. Rank'])
    top5 = df.head(5)

    return df, top5
