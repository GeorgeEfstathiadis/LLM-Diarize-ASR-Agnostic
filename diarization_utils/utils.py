"""Utility functions and classes."""
from collections.abc import Sequence

import numpy as np
from scipy import optimize

"""Function for the Levenshtein algorithm."""
import numpy as np
from enum import Enum


class EditOp(Enum):
  Correct = 0
  Substitution = 1
  Insertion = 2
  Deletion = 3


# Computes the Levenshtein alignment between strings ref and hyp, where the
# tokens in each string are separated by delimiter.
# Outputs a tuple : (edit_distance, alignment) where
# alignment is a list of pairs (ref_pos, hyp_pos) where ref_pos is a position
# in ref and hyp_pos is a position in hyp.
# As an example, for strings 'a b' and 'a c', the output would look like:
# (1, [(0,0), (1,1)]
# Note that insertions are represented as (-1, j) and deletions as (i, -1).
def levenshtein_with_edits(
    ref: str,
    hyp: str,
    delimiter: str = " ",
    print_debug_info: bool = False) -> tuple[int, list[tuple[int, int]]]:
  align = []
  s1 = ref.split(delimiter)
  s2 = hyp.split(delimiter)
  n1 = len(s1)
  n2 = len(s2)
  costs = np.zeros((n1+1, n2+1), dtype=np.int16)
  backptr = np.zeros((n1+1, n2+1), dtype=EditOp)

  for i in range(n1+1):  # ref
    costs[i][0] = i  # deletions

  for j in range(n2):  # hyp
    costs[0][j+1] = j+1  # insertions
    for i in range(n1):  # ref
      # (i,j) <- (i,j-1)
      ins = costs[i+1][j] + 1
      # (i,j) <- (i-1,j)
      del_ = costs[i][j+1] + 1
      # (i,j) <- (i-1,j-1)
      sub = costs[i][j] + (s1[i] != s2[j])
      costs[i + 1][j + 1] = min(ins, del_, sub)
      if (costs[i+1][j+1] == ins):
        backptr[i+1][j+1] = EditOp.Insertion
      elif (costs[i+1][j+1] == del_):
        backptr[i+1][j+1] = EditOp.Deletion
      elif (s1[i] == s2[j]):
        backptr[i+1][j+1] = EditOp.Correct
      else:
        backptr[i+1][j+1] = EditOp.Substitution

  if print_debug_info:
    print("Mincost: ", costs[n1][n2])
  i = n1
  j = n2
  # Emits pairs (n1_pos, n2_pos) where n1_pos is a position in n1 and n2_pos
  # is a position in n2.
  while (i > 0 or j > 0):
    if print_debug_info:
      print("i: ", i, " j: ", j)
    ed_op = EditOp.Correct
    if (i >= 0 and j >= 0):
      ed_op = backptr[i][j]
    if (i >= 0 and j < 0):
      ed_op = EditOp.Deletion
    if (i < 0 and j >= 0):
      ed_op = EditOp.Insertion
    if (i < 0 and j < 0):
      raise RuntimeError("Invalid alignment")
    if (ed_op == EditOp.Insertion):
      align.append((-1, j-1))
      j -= 1
    elif (ed_op == EditOp.Deletion):
      align.append((i-1, -1))
      i -= 1
    else:
      align.append((i-1, j-1))
      i -= 1
      j -= 1

  align.reverse()
  return costs[n1][n2], align


def normalize_text(text: str) -> str:
  """Normalize text."""
  # Convert to lower case.
  text_lower = text.lower()
  # Remove punctuation.
  text_de_punt = (
      text_lower.replace(",", "").replace(".", "").replace("_", "").strip()
  )
  if len(text_lower.split()) == len(text_de_punt.split()):
    return text_de_punt
  else:
    # If ater removing punctuation, we dropped words, then we keep punctuation.
    return text_lower


def get_aligned_hyp_speakers(
    hyp_text: str,
    ref_text: str,
    ref_spk: str,
    print_debug_info: bool = False,
) -> str:
  """Align ref_text to hyp_text, then apply the alignment to ref_spk."""
  # Counters for insertions and deletions in hyp and ref text alignment.
  num_insertions, num_deletions = 0, 0

  # Get the alignment.
  _, align = levenshtein_with_edits(
      normalize_text(ref_text), normalize_text(hyp_text)
  )

  ref_spk_list = ref_spk.split()
  hyp_spk_align = []

  # Apply the alignment on ref speakers.
  for i, j in align:
    if i == -1:
      # hyp has insertion
      hyp_spk_align.append("-1")
      num_insertions += 1
    elif j == -1:
      # hyp has deletion
      num_deletions += 1
      continue
    else:
      hyp_spk_align.append(ref_spk_list[i])
  hyp_spk_align = " ".join(hyp_spk_align)

  if print_debug_info:
    print("Number of insertions: ", num_insertions)
    print("Number of deletions: ", num_deletions)
    # This is not the traditional denominator of WER. Instead, this is
    # len(hyp) + len(ref) - len(SUB).
    print("Length of align pairs: ", len(align))
  return hyp_spk_align


def get_oracle_speakers(hyp_spk: str, hyp_spk_align: str) -> Sequence[int]:
  """Get the oracle speakers for hypothesis."""
  hyp_spk_list = [int(x) for x in hyp_spk.split()]
  hyp_spk_align_list = [int(x) for x in hyp_spk_align.split()]

  # Build cost matrix.
  max_spk = max(max(hyp_spk_list), max(hyp_spk_align_list))
  cost_matrix = np.zeros((max_spk, max_spk))
  for aligned, original in zip(hyp_spk_align_list, hyp_spk_list):
    cost_matrix[aligned - 1, original - 1] += 1

  # Solve alignment.
  row_index, col_index = optimize.linear_sum_assignment(
      cost_matrix, maximize=True
  )

  # Build oracle.
  hyp_spk_oracle = hyp_spk_list.copy()
  for i in range(len(hyp_spk_list)):
    if hyp_spk_align_list[i] == -1:
      # There are some missing words. In such cases, we just use the original
      # speaker for these words if possible.
      if hyp_spk_list[i] == -1:
        # If we don't have original speaker for missing words, just use the
        # previous speaker if possible.
        # This is useful for the update_hyp_text_in_utt_dict() function.
        if i == 0:
          hyp_spk_oracle[i] = 1
        else:
          hyp_spk_oracle[i] = hyp_spk_oracle[i - 1]
      continue
    assert row_index[hyp_spk_align_list[i] - 1] == hyp_spk_align_list[i] - 1
    hyp_spk_oracle[i] = col_index[hyp_spk_align_list[i] - 1] + 1

  return hyp_spk_oracle


# Transcript-Preserving Speaker Transfer (TPST)
def transcript_preserving_speaker_transfer(
    src_text: str, src_spk: str, tgt_text: str, tgt_spk: str
) -> str:
  """Apply source speakers to target."""
  if len(tgt_text.split()) != len(tgt_spk.split()):
    raise ValueError("tgt_text and tgt_spk must have the same length")
  if len(src_text.split()) != len(src_spk.split()):
    raise ValueError("src_text and src_spk must have the same length")
  tgt_spk_align = get_aligned_hyp_speakers(
      hyp_text=tgt_text,
      ref_text=src_text,
      ref_spk=src_spk,
  )
  oracle_speakers = get_oracle_speakers(
      hyp_spk=tgt_spk, hyp_spk_align=tgt_spk_align
  )
  return " ".join([str(x) for x in oracle_speakers])
