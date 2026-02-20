# Investigation Report: save_object() Not Writing Files to Disk

## Summary
The `save_object()` function in `/home/claw/rush-py/src/rush/client.py` has a critical bug that prevents files from being saved to disk when `extract=True` is used. This bug is triggered when `save_energy_outputs()` calls it with extraction enabled.

## Files Involved
- **client.py**: `save_object()` function (lines 417-475)
- **exess.py**: `save_energy_outputs()` function (lines 1069-1107)
- **exess.py**: `chelpg()` function (lines 930-980) - shows working alternative approach

---

## The Bug in save_object() [Lines 450-465 in client.py]

### Current Code:
```python
else:
    data = download_object(path)
    if extract:
        decompressed = zstd.ZstdDecompressor().decompress(
            data, max_output_size=int(1e9)
        )
        with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
            tar_filenames = tar.getnames()
            if len(tar_filenames) >= 2:
                data = tar.extractfile(tar_filenames[1]).read()  # type: ignore
        if len(tar_filenames) >= 2:
            with open(filepath, "wb") as f:
                f.write(data)
    else:
        with open(filepath, "wb") as f:
            f.write(data)
```

### The Problem:
There are **TWO critical issues**:

1. **Silent Failure When Tar Has < 2 Files:**
   - Line 461: `if len(tar_filenames) >= 2:` - extracts file only if >= 2 files
   - Line 464: `if len(tar_filenames) >= 2:` - writes to disk only if >= 2 files
   - **If tar has fewer than 2 files, the extracted data is never written to disk**
   - No error is raised; it silently continues and returns the filepath

2. **No Extraction Fallback:**
   - After decompressing and extracting, if the tar extraction fails or has issues:
     - The original compressed `data` is lost (overwritten)
     - Nothing is written to disk
     - The function returns successfully with an empty/invalid file

### Example Failure Scenario:
```
1. save_object() is called with extract=True
2. download_object() returns binary tar.zst data
3. zstd decompressor extracts successfully
4. tarfile.open() succeeds and gets tar_filenames
5. **if len(tar_filenames) < 2:** CONDITION FAILS
6. data is never reassigned (still contains compressed bytes)
7. **if len(tar_filenames) < 2:** CONDITION FAILS AGAIN
8. File is NEVER written to disk
9. Function returns Path object pointing to non-existent/empty file
```

---

## How It's Called From save_energy_outputs() [Lines 1084-1089]

```python
if "Hdf5" in res[1]:
    hdf5_obj = res[1]["Hdf5"]
    return (
        save_object(res[0]["path"]),
        save_object(
            hdf5_obj["path"],
            ext="hdf5" if extract else "tar.zst",
            extract=extract,  # ← DEFAULT is True (from function signature line 1070)
        ),
    )
```

**Key Issue:** The `extract` parameter defaults to `True` in `save_energy_outputs()` (line 1070), so **every call without explicit `extract=False` triggers the buggy path**.

When called like this:
- Type is inferred as "bin" (not "json")
- `extract=True` is used
- `ext="hdf5"` is set
- The decompression/extraction path is taken, triggering the bug

---

## Why download_object() Works But save_object() Doesn't

### download_object() [client.py, lines 379-397]
```python
def download_object(path: str):
    # Simple, no extraction
    query = gql(...)
    result = _get_client().execute(query)
    obj_descriptor = result["object_path"]
    
    if "contents" in obj_descriptor:
        return obj_descriptor["contents"]
    elif "url" in obj_descriptor:
        response = requests.get(obj_descriptor["url"])
        return response.content
```

**Why it works:** It just downloads and returns bytes. No extraction logic, no conditional writes.

### save_object() Failure
The extraction logic adds complexity that isn't properly handled. The tar archive handling assumes >= 2 files exist, but doesn't validate this or handle the case where extraction fails.

---

## Comparison with chelpg() - Working Alternative [exess.py, lines 930-960]

The `chelpg()` function does NOT use `save_object()` for extraction. Instead, it:

```python
# ✅ WORKING APPROACH
qm_output = download_object(hdf5_obj["path"])
decompressed = zstd.ZstdDecompressor().decompress(
    qm_output, max_output_size=int(1e9)
)
with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
    hdf5_f = tar.extractfile(tar.getnames()[1])  # Gets tar.getnames()[1]
    with h5py.File(hdf5_f, "r") as f:
        # ... processes the file immediately
```

This works because:
1. It calls `download_object()` directly
2. Performs decompression explicitly
3. Extracts tar and uses the file immediately
4. Doesn't rely on conditional file writing logic

---

## What Needs to Be Fixed

### Option 1: Fix save_object() (Recommended)
Refactor the extraction logic to properly handle all cases:

```python
else:
    data = download_object(path)
    if extract:
        # Decompress zstd
        decompressed = zstd.ZstdDecompressor().decompress(
            data, max_output_size=int(1e9)
        )
        # Extract tar
        with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
            tar_filenames = tar.getnames()
            # Validate we have files to extract
            if tar_filenames:  # At least 1 file
                # Prefer 2nd file if available, else use 1st
                tar_idx = min(1, len(tar_filenames) - 1)
                extracted_data = tar.extractfile(tar_filenames[tar_idx]).read()
                data = extracted_data
            else:
                # No files in tar - use original compressed data as fallback
                # OR raise an error
                raise ValueError(f"No files found in tar archive at {path}")
    
    # ALWAYS write (not conditional on tar size)
    with open(filepath, "wb") as f:
        f.write(data)
```

### Option 2: Don't Use save_object() for HDF5 Extraction
In `save_energy_outputs()`, use `download_object()` directly like `chelpg()` does:

```python
if "Hdf5" in res[1]:
    hdf5_obj = res[1]["Hdf5"]
    # First output (JSON)
    json_path = save_object(res[0]["path"])
    
    # Second output (HDF5) - do extraction manually
    if extract:
        hdf5_data = download_object(hdf5_obj["path"])
        decompressed = zstd.ZstdDecompressor().decompress(
            hdf5_data, max_output_size=int(1e9)
        )
        with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
            tar_filenames = tar.getnames()
            extracted = tar.extractfile(tar_filenames[1]).read()
        # Save manually
        filepath = Path(...) / f"{hdf5_obj['path']}.hdf5"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(extracted)
    else:
        hdf5_path = save_object(hdf5_obj["path"], ext="tar.zst", extract=False)
    
    return (json_path, hdf5_path)
```

---

## Root Cause Summary

| Issue | Location | Impact |
|-------|----------|--------|
| **Conditional write on tar file count** | save_object() lines 464-465 | File not written if tar has < 2 files |
| **No fallback for failed extraction** | save_object() line 461-465 | Silent failure, no error raised |
| **Assumption about tar structure** | save_object() line 462 | Assumes tar.getnames()[1] exists |
| **Default extract=True in save_energy_outputs()** | exess.py line 1070 | Always triggers extraction by default |

---

## Dependencies Present
✅ All required imports are present:
- `tarfile` (imported at top of exess.py line 17)
- `zstandard as zstd` (imported at top of exess.py line 19)
- `BytesIO` (imported at top of exess.py line 18)

No missing dependencies are causing silent failures. The bug is pure logic error.

---

## Recommendation
**Fix Option 1 is preferred** because:
1. Makes `save_object()` reliable for all use cases
2. Other code might rely on this function with `extract=True`
3. Creates a general-purpose solution rather than a band-aid

The fix should:
1. Ensure extracted data is written to disk regardless of tar file count
2. Add validation/error handling for tar extraction
3. Handle edge cases where tar is empty or malformed
