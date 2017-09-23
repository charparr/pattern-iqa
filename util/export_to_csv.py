import pandas as pd
from util.zonal_stats import get_zonal_stats


def export_results(top5, outpath):

    df0 = get_zonal_stats(top5.iloc[0]['snakes'], top5.iloc[0]['test_im'], top5.index[0])
    df1 = get_zonal_stats(top5.iloc[0]['snakes'], top5.iloc[1]['test_im'], top5.index[1])
    df2 = get_zonal_stats(top5.iloc[0]['snakes'], top5.iloc[2]['test_im'], top5.index[2])
    df3 = get_zonal_stats(top5.iloc[0]['snakes'], top5.iloc[3]['test_im'], top5.index[3])
    df4 = get_zonal_stats(top5.iloc[0]['snakes'], top5.iloc[4]['test_im'], top5.index[4])
    df = pd.concat([df0, df1, df2, df3, df4], axis=1)
    df['label'] = 'drift'
    df.to_csv(outpath)

    return df


def export_not_drift_results(top5, outpath):

    df0 = get_zonal_stats(top5.iloc[0]['inv_snakes'], top5.iloc[0]['test_im'], top5.index[0])
    df1 = get_zonal_stats(top5.iloc[0]['inv_snakes'], top5.iloc[1]['test_im'], top5.index[1])
    df2 = get_zonal_stats(top5.iloc[0]['inv_snakes'], top5.iloc[2]['test_im'], top5.index[2])
    df3 = get_zonal_stats(top5.iloc[0]['inv_snakes'], top5.iloc[3]['test_im'], top5.index[3])
    df4 = get_zonal_stats(top5.iloc[0]['inv_snakes'], top5.iloc[4]['test_im'], top5.index[4])
    df = pd.concat([df0, df1, df2, df3, df4], axis=1)
    df['label'] = 'not drift'
    df.to_csv(outpath)

    return df