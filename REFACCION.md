# Plan de Refactorización: Migración de Pickle/Dill a Msgpack

Este documento detalla los pasos necesarios para eliminar las vulnerabilidades de ejecución remota de código (RCE) en el proyecto `learned_optimization`, reemplazando el uso inseguro de `pickle` y `dill` por `msgpack` y `msgpack-numpy`.

## 1. Justificación Técnica

El uso de `pickle` y `dill` permite la ejecución de código arbitrario durante la deserialización mediante el método `__reduce__`. Al integrar esto con abstracciones de sistema de archivos que resuelven rutas remotas (como `gs://` o UNC paths), se crea un vector de RCE crítico. `msgpack` es un formato de serialización binario seguro que no ejecuta código durante la reconstrucción de objetos.

## 2. Dependencias Nuevas

Es necesario añadir las siguientes librerías al entorno:
- `msgpack`: Para la serialización base.
- `msgpack-numpy`: Para manejar arreglos de NumPy/JAX de forma eficiente.

**Instalación:**
```bash
pip install msgpack msgpack-numpy
```

## 3. Puntos de Intervención Identificados

### A. Baselines (`learned_optimization/baselines/utils.py`)
- **Problema**: La función `read_npz` usa `onp.load(..., allow_pickle=True)`.
- **Corrección**: 
    1. Cambiar `allow_pickle=False` para forzar seguridad.
    2. Si se requiere guardar metadatos complejos, migrar el formato `.npz` a un archivo `.msgpack` usando `msgpack_numpy`.

### B. Controlador de Población (`learned_optimization/population/population.py`)
- **Problema**: `load_state` usa `pickle.loads(content)` y `serialized_state` usa `pickle.dumps`.
- **Corrección**: Reemplazar por `msgpack.packb` y `msgpack.unpackb` con un codificador personalizado para manejar tipos de JAX/Flax.

### C. Evaluación Continúa (`learned_optimization/continuous_eval/task_group_server.py`)
- **Problema**: El `TaskGroupChief` usa `dill.loads` y `dill.dumps` para persistir el estado de las tareas.
- **Corrección**: Migrar a `msgpack`. Dado que `dill` se usa para objetos más complejos, se debe asegurar que todas las tareas e índices sean tipos serializables básicos o representables mediante diccionarios.

### D. Checkpoints (`learned_optimization/checkpoints.py`)
- **Verificación**: Asegurar que las llamadas a `serialization.from_bytes` de Flax no tengan backends de pickle activos para tipos no registrados.

## 4. Estructura de la Solución Propuesta

Se propone crear un módulo centralizado `learned_optimization/serialization_utils.py`:

```python
import msgpack
import msgpack_numpy as m
import numpy as np

m.patch() # Registra el soporte global de numpy para msgpack

def safe_pack(obj):
    return msgpack.packb(obj, use_bin_type=True)

def safe_unpack(data):
    return msgpack.unpackb(data, raw=False)
```

## 5. Plan de Verificación y Pruebas Unitarias

Se deben implementar las siguientes pruebas para asegurar la robustez de la migración:

### A. Prueba de Integridad de Datos (Funcional)
```python
def test_msgpack_serialization_integrity():
    original_state = {
        "params": np.array([1.0, 2.0], dtype=np.float32),
        "step": 42,
        "metadata": {"task": "mnist", "active": True}
    }
    packed = safe_pack(original_state)
    unpacked = safe_unpack(packed)
    
    assert unpacked["step"] == original_state["step"]
    assert np.array_equal(unpacked["params"], original_state["params"])
    assert unpacked["metadata"]["task"] == "mnist"
```

### B. Prueba de Resistencia a RCE (Seguridad)
```python
def test_msgpack_rejects_pickle_payload():
    import pickle
    class Malicious:
        def __reduce__(self):
            return (os.system, ('echo RCE',))
    
    pickle_payload = pickle.dumps(Malicious())
    
    with pytest.raises(Exception): # Msgpack debe fallar al decodificar una cabecera de pickle
        safe_unpack(pickle_payload)
```

### C. Prueba de Persistencia en Disco
- Verificar que `PopulationController.save_state()` genere un archivo que pueda ser leído por un cliente independiente de `msgpack`.

## 6. Higiene del Código (Code Hygiene)

Todos los archivos modificados durante esta refactorización deben cumplir con los estándares de estilo de Python de Google. Esto se garantiza mediante las siguientes convenciones:
- **isort**: Para mantener una organización consistente y limpia de las importaciones.
- **pyink**: Una variante de `black` adaptada a las convenciones de Google, asegurando un formato de código uniforme y profesional.

---
**Nota**: Este cambio eliminará por completo la superficie de ataque de deserialización insegura reportada, moviendo el proyecto hacia un modelo de "Seguridad por Diseño".
