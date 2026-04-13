## Compliance with CONTRIBUTING.md

Consistent with the project's security and quality guidelines, this PR includes:
- **Unit Tests**: Integrated security validation logic in `learned_optimization/serialization_test.py` to ensure data integrity and resistance to deserialization attacks.
- **Verified Fix**: Successfully reproduced RCE vulnerabilities using malicious `__reduce__` payloads in `.npz` archives and state files, then verified their neutralization via the new `msgpack` serialization layer and hardened NumPy loading gates.
- **Code Hygiene**: All modified files adhere to Google's standard Python style (enforced via `isort` and `pyink` conventions).

---

## Description

This PR addresses critical security vulnerabilities in the `learned_optimization` library related to insecure data deserialization. The existing implementation utilized `pickle` and `dill` for persisting population states, baseline archives, and task group metadata. These formats are inherently insecure when loading data from untrusted or remote sources (e.g., GCS, UNC paths), as they allow for arbitrary Remote Code Execution (RCE).

This refactoring transitions the framework to a **"Secure by Default"** architecture by replacing `pickle` and `dill` with **Msgpack** (a data-only binary format) for all automated serialization tasks. This change ensures that the infrastructure-level security is improved without compromising the core functionality of the optimization loops.

### Key Security Improvements:

1.  **Elimination of Insecure Deserialization in Population Management**: Replaced `pickle` with `msgpack` in `PopulationController`. By using a non-executable binary format, we prevent potential system takeovers when restoring population states from shared or remote filesystems.
2.  **Baselines Hardening**: Updated the baseline loading utility to disable `allow_pickle` in `numpy.load`. This closes a primary RCE vector where attackers could ship malicious `.npz` files disguised as baseline archives.
3.  **Secure State Persistence for Continuous Eval**: Refactored the `TaskGroupChief` to use `msgpack` instead of `dill` for internal state restoration, eliminating another critical sink used in distributed training environments.
4.  **Centralized Serialization Utility**: Introduced a dedicated `serialization_utils` module that provides safe `pack` and `unpack` primitives, facilitating future security audits and consistent behavior across the project.

## Technical Implementation Details

- **`learned_optimization/serialization_utils.py`**:
    - New module implementing `safe_pack` and `safe_unpack` using `msgpack` and `msgpack-numpy`.
    - Includes a custom encoder to safely handle NumPy arrays and standard Python objects without allowing arbitrary class instantiation.
- **`learned_optimization/population/population.py`**:
    - Refactored `load_state` and `serialized_state` to use the new `serialization_utils`.
- **`learned_optimization/baselines/utils.py`**:
    - Hardened `read_npz` by setting `allow_pickle=False`. Added documentation warning about the risks of legacy pickle loading.
- **`learned_optimization/continuous_eval/task_group_server.py`**:
    - Migrated away from `dill` to `msgpack` for task state persistence, ensuring inter-process safety.
- **`requirements.txt`**:
    - Added `msgpack>=1.0.0` and `msgpack-numpy>=0.4.7` as core dependencies.

## Verification Performed

- Developed and executed a comprehensive security suite (**`learned_optimization/serialization_test.py`**) that confirms:
    - **Neutralization**: Malicious `pickle` payloads now trigger a decoding error instead of executing code.
    - **Data Integrity**: Verified that NumPy arrays and nested dictionaries are perfectly preserved during the transition to `msgpack`.
    - **Functional Parity**: Confirmed that the population controller can successfully save and restore its state using the new binary format.
