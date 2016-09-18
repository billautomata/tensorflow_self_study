# Identify encrypted data
Create two sets of data, one encrypted values from inputs of bytes with values from `0-255`, and another set of bytes using a restricted range `65-90`, meant to simulate characters in the ASCII range.  All of the data is encoded to base64.


```
encrypted value, is_restricted
"AEUDNEK...", "yes"
```

Build and train a classifier that tries to identify encrypted values created from the restricted range.
