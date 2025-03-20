


```
 messages = [
        {"role": "system", "content": ". Given scene content and user edit request, determine the appropriate transformation matrix.\n\n"
                                    "Scene content: " + scene_context + "\n\n"
                                    "Available transformations:\n"
                                    "1. Translation [[1 0 tx][0 1 ty][0 0 1]]\n"
                                    "2. Rotation [[cos(θ) -sin(θ) 0][sin(θ) cos(θ) 0][0 0 1]]\n"
                                    "3. Scale [[sx 0 0][0 sy 0][0 0 1]]\n"
                                    "4. Shear [[1 shx 0][shy 1 0][0 0 1]]\n\n"
                                    "Combined transformations are allowed by matrix multiplication.\n\n"
                                    "IMPORTANT: Your response MUST include the final transformation matrix after the token '$answer$' in this exact format:\n"
                                    "$answer$\n"
                                    "[[0.88 0.  0. ]\n"
                                    " [0.  0.88 0. ]\n"
                                    " [0.  0.  1. ]]"},
        {"role": "user", "content": user_edit}
    ]
```

```
messages = [
        {"role": "system", "content": "Integrate natural language reasoning with programs to solve user query. Given the scene content and the user edit, determine the appropriate transformation matrix for the requested edit.\n\n"
                                    "Scene content: " + scene_context + "\n\n"
                                    "List of possible operations:\n"
                                    "1. Translation: Moving objects in x,y directions\n"
                                    "   Example: [[1 0 tx][0 1 ty][0 0 1]]\n\n"
                                    "2. Rotation: Rotating objects by angle θ\n"
                                    "   Example: [[cos(θ) -sin(θ) 0][sin(θ) cos(θ) 0][0 0 1]]\n\n"
                                    "3. Scaling: Changing object size\n"
                                    "   Example: [[sx 0 0][0 sy 0][0 0 1]]\n\n"
                                    "4. Shear: Skewing objects\n"
                                    "   Example: [[1 shx 0][shy 1 0][0 0 1]]\n\n"
                                    "5. Combined transformations are also allowed:\n"
                                    "   Example: multiply the transformation matrices corresponding to the operations. for example translation + rotation = [[cos(θ) -sin(θ) tx][sin(θ) cos(θ) ty][0 0 1]] * [[1 0 tx][0 1 ty][0 0 1]]; additional examples: translation + scaling = [[1 0 tx][0 1 ty][0 0 1]] * [[sx 0 0][0 sy 0][0 0 1]], translation + rotation + scaling = [[cos(θ) -sin(θ) tx][sin(θ) cos(θ) ty][0 0 1]] * [[sx 0 0][0 sy 0][0 0 1]]   \n\n"
                                    " NOTE: I NEED THE FINAL MATRIX as a numpy array after a word token \"$answer$\" , for example: $answer$\n"
                                    "[[0.88 0.  0. ]\n"
                                    " [0.  0.88 0. ]\n"
                                    " [0.  0.  1. ]]  \n\n"},
        {"role": "user", "content": user_edit}
    ]
```
