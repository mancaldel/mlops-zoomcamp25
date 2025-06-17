#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Use a trained model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--save_results', action="store_true", default=False, help='Type of model to train')
    args = parser.parse_args()

    print("Predicting:")
    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year:04d}-{args.month:02d}.parquet'
    print(f"- Loading '{data_url}'...")
    df = read_data(data_url)


    print("- Transforming and predicting...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')


    df_result = df[["ride_id"]]
    df_result["prediction"] = y_pred
    output_file = "prediction.parquet"

    # Save results
    if args.save_results:
        df_result.to_parquet(
            output_file,
            engine='pyarrow',
            compression=None,
            index=False
        )

    # Print mean and std
    print("Results:")
    print(f"- Duration mean: {y_pred.mean():8.3f}")
    print(f"- Duration std:  {y_pred.std():8.3f}")
