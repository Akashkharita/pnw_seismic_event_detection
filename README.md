## Using the trained **QuakeXNet** model with **SeisBench**

### 1) Install SeisBench

Install SeisBench in your environment (example via pip):

```bash
pip install seisbench
```

---

### 2) Add the custom model file to SeisBench

Copy your model definition into SeisBench’s models package:

```bash
cp src/quakexnet.py <PATH_TO_SITE_PACKAGES>/seisbench/models/quakexnet.py
```

> You should copy `src/quakexnet.py` into:
> `seisbench/seisbench/models/` (i.e., the installed package directory)

---

### 3) Register the model in `seisbench.models`

Open the following file:

```
<PATH_TO_SITE_PACKAGES>/seisbench/models/__init__.py
```

Add this line:

```python
from .quakexnet import QuakeXNet
```

---

### 4) Add the trained weights to the SeisBench cache directory

Create the target directory (if it doesn’t exist):

```bash
mkdir -p ~/.seisbench/models/v3/quakexnet
```

Copy the trained weights:

```bash
cp src/base.pt.v3 ~/.seisbench/models/v3/quakexnet/base.pt.v3
```

---

### 5) Create the minimal model metadata file

In the same directory (`~/.seisbench/models/v3/quakexnet`), create `base.json.v3`:

```bash
echo '{}' > ~/.seisbench/models/v3/quakexnet/base.json.v3
```

Your directory should look like:

```text
~/.seisbench/models/v3/quakexnet/
├── base.pt.v3
└── base.json.v3
```

---

### 6) Import and use the model

Once the model is registered, you should be able to import it like:

```python
import seisbench.models as sbm

model = sbm.QuakeXNet()
```
