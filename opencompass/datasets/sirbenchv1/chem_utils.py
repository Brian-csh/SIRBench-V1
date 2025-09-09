from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import difflib

def get_scaffold_fp(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
    except Exception as e:
        scaffold_fp = None   # 修改，让他处理一下异常
    return scaffold_fp

def get_scaffold_fp(x):
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
    except Exception as e:
        scaffold_fp = None   # 修改，让他处理一下异常
    return scaffold_fp

def top_n_scaffold_similar_molecules(target_smiles, molecule_scaffold_list, molecule_smiles_list, n=5):
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)

    similarities = []

    for idx, scaffold_fp in enumerate(molecule_scaffold_list):
        try:
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            similarities.append((idx, tanimoto_similarity))
        except Exception as e:
            # print(e)
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_similar_molecules = similarities[:n]

    return [molecule_smiles_list[i[0]] for i in top_5_similar_molecules]

def similarity_ratio(s1, s2):
    # Calculate the similarity ratio between the two strings
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

    # Return the similarity ratio
    return ratio

def top_n_similar_strings(query, candidates, n=5):
    # Calculate the Levenshtein distance between the query and each candidate
    distances = [(c, similarity_ratio(query, c)) for c in candidates]

    # Sort the candidates by their Levenshtein distance to the query
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)

    # Get the top n candidates with the smallest Levenshtein distance
    top_candidates = [d[0] for d in sorted_distances[:n]]

    # Return the top n candidates
    return top_candidates

def molToCanonical(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles
    except:
        return None