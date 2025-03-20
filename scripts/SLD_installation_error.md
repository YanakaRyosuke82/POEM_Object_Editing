
## 📑 Appendix
### ⚠️ ERRORS at installation of SLD:
```
modified:   models/attention.py
modified:   models/attention_processor.py
modified:   models/unet_2d_blocks.py
```

0.
```
  from diffusers.models.dual_transformer_2d import DualTransformer2DModel
  # at SLD/models/unet_2d_blocks.py
 ```
change it to
``` from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel ```


1.
```
ImportError: cannot import name 'maybe_allow_in_graph' from 'diffusers.utils' (/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/.venv2/lib/python3.10/site-packages/diffusers/utils/__init__.py)
```
change it to ``` from diffusers.utils.torch_utils import maybe_allow_in_graph ``` or just comment out.



2.
```
 File "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/ext_module/SLD/models/attention_processor.py", line 21, in <module>
    from diffusers.utils import deprecate, logging, maybe_allow_in_graph
ImportError: cannot import name 'maybe_allow_in_graph' from 'diffusers.utils'
```
just comment out



3.
```
 File "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/ext_module/SLD/models/models.py", line 5, in <module>
    from easydict import EasyDict
ModuleNotFoundError: No module named 'easydict'
```
fix with    ```pip install easydict```


4.
```
  File "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/ext_module/SLD/utils/parse.py", line 7, in <module>
    import inflect
ModuleNotFoundError: No module named 'inflect'
```


5.  File "/dtu/blackhole/14/189044/marscho/VLM_controller_for_SD/ext_module/SLD/sld/llm_chat.py", line 3, in <module>
    from openai import OpenAI
ModuleNotFoundError: No module named 'openai'
