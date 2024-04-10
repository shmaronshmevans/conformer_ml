import os


def make_dirs(today_date, station):
    if (
        os.path.exists(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}")
        == False
    ):
        os.mkdir(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}")
        os.mkdir(f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}")
    if (
        os.path.exists(
            f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{station}"
        )
        == False
    ):
        os.mkdir(f"/home/aevans/transformer_ml/src/data/visuals/{today_date}/{station}")
        os.mkdir(f"/home/aevans/transformer_ml/src/data/temp_df/{today_date}/{station}")
