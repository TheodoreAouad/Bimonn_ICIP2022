from sklearn.model_selection import train_test_split


def train_val_test_split(*arrays, train_size=None, val_size=None, test_size=None, random_state=None, shuffle=True, stratify=None):
    test_split = []

    train_split, val_split = train_test_split(
        *arrays, test_size=test_size+val_size, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify
    )
    if test_size is not None and test_size > 0:
        val_split, test_split = train_test_split(
            val_split, train_size=val_size, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify
        )

    return train_split, val_split, test_split
