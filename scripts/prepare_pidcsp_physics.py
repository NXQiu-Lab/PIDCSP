import os
import torch
import pandas as pd
from mendeleev.fetch import fetch_ionization_energies, fetch_table

def prepare_physical_properties(save_path="embeddings/phys_embedding_data.pt"):


    properties_list = [
        "atomic_radius",
        "atomic_volume",
        "density",
        "dipole_polarizability",
        "electron_affinity",
        "en_allen",
        "vdw_radius",
        "metallic_radius",
        "metallic_radius_c12",
        "covalent_radius_pyykko_double",
        "covalent_radius_pyykko_triple",
        "covalent_radius_pyykko",
        "IE1",
        "IE2",
    ]

    print("Fetching element data from mendeleev...")


    df = fetch_table("elements")
    df = df.set_index("atomic_number")


    ies = fetch_ionization_energies(degree=[1, 2])
    df = pd.concat([df, ies], axis=1)


    df.group_id = df.group_id.fillna(value=19.0)
    group = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
    ])

    period = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.tensor(df.period.loc[:100].values, dtype=torch.long),
    ])

    group_size = int(df.group_id.unique().shape[0])
    period_size = int(df.period.loc[:100].unique().shape[0])


    df_props = df[properties_list].loc[:85, :]


    props_mean = df_props.mean()
    props_std = df_props.std()
    df_props = (df_props - props_mean) / props_std


    valid_cols = df_props.columns[pd.isnull(df_props).sum() < int(0.5 * df_props.shape[0])].tolist()
    df_props = df_props[valid_cols]


    col_missing = df_props.columns[df_props.isna().any()].tolist()
    df_props[col_missing] = df_props[col_missing].fillna(df_props[col_missing].mean())

    n_properties = len(df_props.columns)


    properties = torch.cat([
        torch.zeros(1, n_properties),
        torch.from_numpy(df_props.values).float(),
    ])


    data = {
        "properties": properties,
        "group": group,
        "period": period,
        "n_properties": n_properties,
        "group_size": group_size,
        "period_size": period_size,
        "properties_list": valid_cols,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)

    print(f"Saved physical embedding data to {save_path}")
    print(f"  - Properties shape: {properties.shape}")
    print(f"  - Group shape: {group.shape}, size: {group_size}")
    print(f"  - Period shape: {period.shape}, size: {period_size}")
    print(f"  - Valid properties: {valid_cols}")

    return data


if __name__ == "__main__":
    prepare_physical_properties()
