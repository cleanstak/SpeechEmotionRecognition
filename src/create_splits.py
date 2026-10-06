from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# SPEAKER-INDEPENDENT DATASET SPLITTING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "ravdess_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_split.csv"
)


# ------------------------------------------------------------
# 1. Load metadata
# ------------------------------------------------------------

df = pd.read_csv(METADATA_FILE)

print(f"Total records loaded: {len(df)}")
print(f"Total actors: {df['actor_id'].nunique()}")


# ------------------------------------------------------------
# 2. First split:
#    70% training
#    30% temporary
#
#    IMPORTANT:
#    Groups = actor_id
#    Therefore an actor cannot appear in both sets.
# ------------------------------------------------------------

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_indices, temp_indices = next(
    splitter_1.split(
        df,
        groups=df["actor_id"]
    )
)

train_df = df.iloc[train_indices].copy()
temp_df = df.iloc[temp_indices].copy()


# ------------------------------------------------------------
# 3. Second split:
#    Temporary → validation + test
#
#    50% validation
#    50% test
#
#    This gives approximately:
#    70% training
#    15% validation
#    15% testing
# ------------------------------------------------------------

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

validation_indices, test_indices = next(
    splitter_2.split(
        temp_df,
        groups=temp_df["actor_id"]
    )
)

validation_df = temp_df.iloc[validation_indices].copy()
test_df = temp_df.iloc[test_indices].copy()


# ------------------------------------------------------------
# 4. Add split labels
# ------------------------------------------------------------

train_df["split"] = "train"
validation_df["split"] = "validation"
test_df["split"] = "test"


# ------------------------------------------------------------
# 5. Combine everything
# ------------------------------------------------------------

final_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)


# Sort for easier inspection
final_df = final_df.sort_values(
    by=["split", "actor_id", "emotion"]
).reset_index(drop=True)


# Save
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFICATION
# ============================================================

print("\nDataset split created successfully.")

print("\nNumber of files in each split:")
print(final_df["split"].value_counts())


print("\nNumber of actors in each split:")
print(
    final_df.groupby("split")["actor_id"]
    .nunique()
)


print("\nActors in each split:")

for split_name in ["train", "validation", "test"]:

    actors = sorted(
        final_df.loc[
            final_df["split"] == split_name,
            "actor_id"
        ].unique()
    )

    print(f"{split_name}: {actors}")


print("\nEmotion distribution by split:")
print(
    pd.crosstab(
        final_df["split"],
        final_df["emotion"]
    )
)


# ------------------------------------------------------------
# Verify that no actor appears in multiple splits
# ------------------------------------------------------------

actor_split_counts = (
    final_df.groupby("actor_id")["split"]
    .nunique()
)

if actor_split_counts.max() > 1:

    print(
        "\nERROR: An actor appears in multiple splits!"
    )

else:

    print(
        "\nSUCCESS: No actor appears in more than one split."
    )


print(f"\nSaved to:")
print(OUTPUT_FILE)