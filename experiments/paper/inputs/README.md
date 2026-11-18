# Input Indices

`final_dataset_manifest.csv` and `final_dataset_tree.txt` are the supplied logical
dataset indices, preserved byte-for-byte. They contain pseudonymous case IDs,
shapes, original split labels and logical destination paths, not MRI arrays.

Private planning DOCX/prompt files are kept local and ignored by Git. They are
not needed to run the checked-in configurations. Re-generating configurations
with generate_configs.py requires the user's complete local planning/input folder.

The optional source-file mapping is also private: provide it locally to
build_virtual_dataset.py. Its copy, resolved indices and all derived caches and
outputs remain under the Git-ignored outputs/paper/virtual_dataset directory.
