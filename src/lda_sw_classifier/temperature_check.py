# import time

# import numpy as np
# import pandas as pd
# import pynmrstar


# # ============================================================
# # Configuration
# # ============================================================

# TARGET_TEMPERATURE_K = 298.0
# TEMPERATURE_TOLERANCE = 1e-6

# MAX_RETRIES = 3
# RETRY_DELAY_SECONDS = 2
# REQUEST_DELAY_SECONDS = 0.2

# OUTPUT_CSV = "bmrb_temperature_results.csv"
# ERRORS_CSV = "bmrb_temperature_errors.csv"


# # ============================================================
# # BMRB IDs: original list + additional IDs
# # ============================================================

# original_ids = [
#     15298, 16876, 19114, 19332, 25384, 26027, 26935, 27868,
#     27964, 30389, 50256, 50274, 50542, 50593, 50896, 51015,
#     51033, 51114, 51116, 51173, 51303, 51449, 51523, 51952,
#     51953, 52292, 52370, 52554, 7244, 11511, 11593, 15141,
#     15506, 16166, 16300, 18177, 18630, 19644, 19648, 25196,
#     26654, 26719, 26728, 26755, 27537, 27618, 27883, 27992,
#     50115, 50189, 50204, 50314, 50326, 50370, 50427, 50437,
#     50438, 50477, 50478, 50558, 50618, 50619, 50687, 50739,
#     50740, 50746, 50758, 50759, 50766, 50795, 50859, 50860,
#     50861, 50862, 50864, 50866, 50888, 50904, 50908, 50916,
#     51018, 51019, 51051, 51062, 51067, 51068, 51069, 51080,
#     51092, 51111, 51112, 51113, 51115, 51120, 51121, 51141,
#     51152, 51178, 51186, 51203, 51204, 51205, 51206, 51207,
#     51208, 51229, 51262, 51282, 51304, 51316, 51319, 51322,
#     51327, 51334, 51376, 51381, 51396, 51427, 51475, 51479,
#     51507, 51534, 51550, 51598, 51601, 51617, 51625, 51635,
#     51636, 51638, 51648, 51682, 51695, 51711, 51717, 51758,
#     51763, 51780, 51782, 51795, 51827, 51828, 51829, 51831,
#     51881, 51899, 51900, 51904, 52016, 52028, 52109, 52178,
#     52181, 52182, 52184, 52207, 52233, 52258, 52269, 52284,
#     52322, 52323, 52324, 52326, 52349, 52379, 52387, 52398,
#     52492, 52503, 52524, 52550, 52551,
# ]

# additional_ids = [
#     4922, 6436, 6968, 15176, 15180, 15719, 16296, 16445,
#     16450, 17290, 17483, 17836, 18417, 19246, 19999, 25044,
#     25118, 25426, 25921, 26530, 26544, 26740, 26770, 26841,
#     27179, 27351, 27426, 27553, 27554, 27596, 27597, 27608,
#     27615, 27626, 27867, 27900, 27962, 28006, 30031, 30271,
#     30390, 30391, 30513, 30599, 30633, 30687, 30773, 30909,
#     31150, 34040, 34582, 34751, 34814, 34847, 36165, 50154,
#     50196, 50201, 50465, 50496, 50497, 50525, 50752, 50803,
#     50944, 50961, 50962, 51011, 51147, 51172, 51175, 51286,
#     51301, 51323, 51353, 51354, 51355, 51364, 51405, 51406,
#     51487, 51522, 51720, 51730, 51888, 51978, 51979, 52130,
#     52141, 52356, 52418, 52525,
# ]

# # Preserve order and remove duplicate IDs.
# bmrb_ids = list(dict.fromkeys(original_ids + additional_ids))


# # ============================================================
# # Helpers for NMR-STAR values
# # ============================================================

# def _clean_nmrstar_value(value):
#     """Normalize NMR-STAR values and convert missing markers to None."""
#     if value is None:
#         return None

#     value = str(value).strip()

#     if value in {"", ".", "?"}:
#         return None

#     return value


# # ============================================================
# # Temperature extraction
# # ============================================================

# def get_temperature_for_shift_list(ent, shift_frame):
#     """
#     Return the experimental temperature in Kelvin for a chemical-shift
#     saveframe, or None if it cannot be established.

#     The temperature is matched using Sample_condition_list_ID.
#     """
#     shift_list_ids = shift_frame.get_tag(
#         "_Assigned_chem_shift_list.Sample_condition_list_ID"
#     )

#     if not shift_list_ids:
#         return None

#     condition_id = _clean_nmrstar_value(shift_list_ids[0])

#     if condition_id is None:
#         return None

#     for loop in ent.get_loops_by_category(
#         "Sample_condition_variable"
#     ):
#         condition_ids = loop.get_tag(
#             "_Sample_condition_variable.Sample_condition_list_ID"
#         )
#         types = loop.get_tag(
#             "_Sample_condition_variable.Type"
#         )
#         values = loop.get_tag(
#             "_Sample_condition_variable.Val"
#         )
#         units = loop.get_tag(
#             "_Sample_condition_variable.Val_units"
#         )

#         for cid, variable_type, value, unit in zip(
#             condition_ids, types, values, units
#         ):
#             if _clean_nmrstar_value(cid) != condition_id:
#                 continue

#             variable_type = _clean_nmrstar_value(variable_type)

#             if variable_type is None:
#                 continue

#             if "temp" not in variable_type.lower():
#                 continue

#             value = _clean_nmrstar_value(value)
#             unit = _clean_nmrstar_value(unit)

#             if value is None or unit is None:
#                 continue

#             try:
#                 temperature = float(value)
#             except (ValueError, TypeError):
#                 continue

#             unit_normalized = unit.strip().lower()

#             if unit_normalized in {"k", "kelvin", "kelvins"}:
#                 return temperature

#             if unit_normalized in {
#                 "c",
#                 "°c",
#                 "deg c",
#                 "degree celsius",
#                 "degrees celsius",
#                 "celsius",
#             }:
#                 return temperature + 273.15

#     return None


# # ============================================================
# # Check one BMRB entry
# # ============================================================

# def check_bmrb_temperature(bmrb_id):
#     """
#     Check all assigned chemical-shift saveframes for a BMRB entry.

#     Returns one result row per BMRB entry.
#     """
#     ent = pynmrstar.Entry.from_database(str(bmrb_id))

#     shift_frames = ent.get_saveframes_by_category(
#         "assigned_chemical_shifts"
#     )

#     if not shift_frames:
#         return {
#             "BMRB_ID": bmrb_id,
#             "n_shift_lists": 0,
#             "temperatures_K": "",
#             "is_298K": False,
#             "status": "No assigned chemical shifts",
#         }

#     temperatures = []
#     missing_temperature_count = 0

#     for shift_frame in shift_frames:
#         temperature = get_temperature_for_shift_list(
#             ent,
#             shift_frame,
#         )

#         if temperature is None or pd.isna(temperature):
#             missing_temperature_count += 1
#             continue

#         temperatures.append(float(temperature))

#     unique_temperatures = sorted(set(temperatures))

#     is_298K = any(
#         np.isclose(
#             temperature,
#             TARGET_TEMPERATURE_K,
#             rtol=0,
#             atol=TEMPERATURE_TOLERANCE,
#         )
#         for temperature in unique_temperatures
#     )

#     if not temperatures:
#         status = "Temperature unavailable"
#     elif missing_temperature_count:
#         status = "Partial: some shift lists lack temperature"
#     else:
#         status = "OK"

#     return {
#         "BMRB_ID": bmrb_id,
#         "n_shift_lists": len(shift_frames),
#         "temperatures_K": "; ".join(
#             f"{temperature:g}"
#             for temperature in unique_temperatures
#         ),
#         "is_298K": is_298K,
#         "status": status,
#     }


# # ============================================================
# # Main
# # ============================================================

# def main():
#     results = []
#     errors = []

#     total = len(bmrb_ids)

#     print(f"Original IDs: {len(original_ids)}")
#     print(f"Additional IDs: {len(additional_ids)}")
#     print(f"Unique entries to check: {total}")
#     print(f"Target temperature: {TARGET_TEMPERATURE_K:g} K\n")

#     for i, bmrb_id in enumerate(bmrb_ids, start=1):
#         print(
#             f"[{i}/{total}] Checking BMRB {bmrb_id}...",
#             flush=True,
#         )

#         for attempt in range(1, MAX_RETRIES + 1):
#             try:
#                 result = check_bmrb_temperature(bmrb_id)
#                 results.append(result)

#                 print(
#                     f"    Temperature(s): "
#                     f"{result['temperatures_K'] or 'unknown'}; "
#                     f"298 K: {result['is_298K']}; "
#                     f"status: {result['status']}",
#                     flush=True,
#                 )
#                 break

#             except Exception as exc:
#                 if attempt < MAX_RETRIES:
#                     print(
#                         f"    Attempt {attempt} failed: "
#                         f"{type(exc).__name__}: {exc}. "
#                         f"Retrying in {RETRY_DELAY_SECONDS}s...",
#                         flush=True,
#                     )
#                     time.sleep(RETRY_DELAY_SECONDS)
#                 else:
#                     errors.append({
#                         "BMRB_ID": bmrb_id,
#                         "error_type": type(exc).__name__,
#                         "error": str(exc),
#                     })

#                     print(
#                         f"    FAILED after {MAX_RETRIES} attempts: "
#                         f"{type(exc).__name__}: {exc}",
#                         flush=True,
#                     )

#         time.sleep(REQUEST_DELAY_SECONDS)

#     # Explicit columns prevent KeyError if results is empty.
#     result_columns = [
#         "BMRB_ID",
#         "n_shift_lists",
#         "temperatures_K",
#         "is_298K",
#         "status",
#     ]

#     df = pd.DataFrame(results, columns=result_columns)
#     df.to_csv(OUTPUT_CSV, index=False)

#     error_columns = ["BMRB_ID", "error_type", "error"]
#     errors_df = pd.DataFrame(errors, columns=error_columns)
#     errors_df.to_csv(ERRORS_CSV, index=False)

#     print("\n" + "=" * 65)
#     print("SUMMARY")
#     print("=" * 65)
#     print(f"Requested unique entries: {total}")
#     print(f"Successfully processed: {len(df)}")
#     print(f"Failed entries: {len(errors)}")

#     if df.empty:
#         print("\nNo entries were successfully processed.")
#         print(f"Inspect: {ERRORS_CSV}")
#         return

#     # Entries with at least one shift list at 298 K.
#     matches = df.loc[
#         df["is_298K"].fillna(False),
#         "BMRB_ID",
#     ].astype(int).tolist()

#     # Entries with recorded temperatures, none of which is 298 K.
#     has_temperature = (
#         df["temperatures_K"].notna()
#         & df["temperatures_K"].ne("")
#     )

#     different_temperature = df.loc[
#         has_temperature
#         & ~df["is_298K"].fillna(False),
#         "BMRB_ID",
#     ].astype(int).tolist()

#     different_temperature_details = df.loc[
#         has_temperature
#         & ~df["is_298K"].fillna(False),
#         ["BMRB_ID", "temperatures_K"],
#     ]

#     # Entries for which no temperature could be established.
#     unavailable = df.loc[
#         df["status"] == "Temperature unavailable",
#         "BMRB_ID",
#     ].astype(int).tolist()

#     no_shifts = df.loc[
#         df["status"] == "No assigned chemical shifts",
#         "BMRB_ID",
#     ].astype(int).tolist()

#     partial = df.loc[
#         df["status"] == "Partial: some shift lists lack temperature",
#         "BMRB_ID",
#     ].astype(int).tolist()

#     # Report entries with temperature 298 K.
#     print(
#         f"\nEntries with temperature {TARGET_TEMPERATURE_K:g} K: "
#         f"{len(matches)}"
#     )
#     print(matches)

#     # Report entries with temperatures other than 298 K.
#     print(
#         f"\nEntries with temperature different from "
#         f"{TARGET_TEMPERATURE_K:g} K: "
#         f"{len(different_temperature)}"
#     )
#     print(different_temperature)

#     print("\nTheir recorded temperatures:")
#     if different_temperature_details.empty:
#         print("None")
#     else:
#         print(different_temperature_details.to_string(index=False))

#     print("\nEntries with unavailable temperatures:")
#     print(unavailable)

#     print("\nEntries with no assigned chemical shifts:")
#     print(no_shifts)

#     print("\nEntries with partially missing temperatures:")
#     print(partial)

#     print(f"\nFull results saved to: {OUTPUT_CSV}")
#     print(f"Errors saved to: {ERRORS_CSV}")


# if __name__ == "__main__":
#     main()

from pathlib import Path

matches_298 = [
19332, 26027, 26935, 27868, 30389, 50542, 51015, 51114, 51523, 52370,
15141, 16166, 18630, 27537, 27883, 50115, 50189, 50314, 50427, 50437,
50438, 50558, 50618, 50619, 50740, 50766, 50795, 50866, 51018, 51019,
51051, 51067, 51068, 51069, 51080, 51092, 51111, 51112, 51113, 51186,
51203, 51262, 51282, 51304, 51427, 51507, 51829, 51899, 52178, 52181,
52182, 52184, 52207, 52258, 52284, 52387, 52398, 52503, 52524, 6436,
15176, 15180, 16296, 17290, 17483, 17836, 18417, 25426, 25921, 26740,
27426, 27596, 27608, 27962, 30271, 30390, 30391, 30599, 30687, 30773,
31150, 34847, 50154, 50196, 50496, 50803, 50961, 50962, 51147, 51172,
51175, 51286, 51353, 51354, 51355, 51487, 51730, 51888, 51979, 52130,
52525
]

different_temp = [
15298, 16876, 19114, 25384, 27964, 50256, 50274, 50593, 50896, 51033,
51116, 51173, 51303, 51449, 51952, 51953, 52292, 52554, 7244, 11511,
11593, 15506, 16300, 18177, 19644, 19648, 25196, 26654, 26719, 26728,
26755, 27618, 27992, 50204, 50326, 50370, 50477, 50478, 50687, 50739,
50746, 50758, 50759, 50859, 50860, 50861, 50862, 50864, 50888, 50904,
50908, 50916, 51062, 51115, 51120, 51121, 51141, 51152, 51178, 51204,
51205, 51206, 51207, 51208, 51229, 51316, 51319, 51322, 51327, 51376,
51381, 51396, 51475, 51479, 51534, 51550, 51598, 51601, 51617, 51625,
51635, 51636, 51638, 51648, 51682, 51695, 51711, 51717, 51758, 51763,
51780, 51782, 51795, 51827, 51828, 51831, 51881, 51900, 51904, 52016,
52028, 52109, 52269, 52322, 52323, 52324, 52326, 52349, 52379, 52492,
52550, 52551, 4922, 6968, 15719, 16445, 16450, 19246, 19999, 25044,
25118, 26530, 26544, 26770, 26841, 27179, 27351, 27553, 27554, 27597,
27615, 27626, 27867, 27900, 28006, 30031, 30513, 30633, 30909, 34040,
34582, 34751, 34814, 36165, 50201, 50465, 50497, 50525, 50752, 50944,
51011, 51301, 51323, 51364, 51405, 51406, 51522, 51720, 51978, 52141,
52356, 52418
]

# Create two newline-delimited TXT files in the active runtime.
out_dir = Path("/home/zosia/Desktop/LDA_SW_tests/combined_lda_sw_classifier/Input")
file_298 = out_dir / "bmrb_temperature_298K.txt"
file_other = out_dir / "bmrb_temperature_different_from_298K.txt"

file_298.write_text("".join(f"{x}\n" for x in matches_298), encoding="utf-8")
file_other.write_text("".join(f"{x}\n" for x in different_temp), encoding="utf-8")

print(f"298 K IDs: {len(matches_298)}; file: {file_298}; exists={file_298.exists()}")
print(f"Other-temperature IDs: {len(different_temp)}; file: {file_other}; exists={file_other.exists()}")
