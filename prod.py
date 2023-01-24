# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import banana_dev as banana

api_key = "<YOUR_API_KEY>"
model_key = "<YOU_MODLE_KEY>"

model_inputs = {'prompt': 'My friends are cool but they eat too many carbs.'}

out = banana.run(api_key, model_key, model_inputs)

print(out)