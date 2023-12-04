# REGEX LIBRARY GAUDI

## Escape Codes #0

Matches escape codes such as \\x0d, \\x0a, etc.
```python
r"(\\[xu][0-9A-Fa-f]{2,})|\\[stepr]", " "
```
**Example:**
```python
# Input String 
'\\x0dTesting escape codes\\x0d\\x0a\\x0d \\r30  '

# Result
' Testing escape codes     30  '
```
<br>

##  Whitespaces

```python
 r'[\r\n\t]|\s{2,}', " "
```
- matches any new line, carriage return tab



### 2 get chosen punctuation

```python
[r'[\\\_,\(\);\[\]#{}\*"\'\~\?!\|\^]', " "]
```
- replace \ _ , ( ) ; [ ] # { } ~ ? ! * ^


### 3 get angle brackets regex

```python
[r'<(.*?)>', r'\1']
```
- matches and keeps the content between the angle brackets


### 4 get percent sign

```python
[r'%', " percent " ]
```
- keep 


### 5 get multiple punctuation

```python
r'[\-\.:\/]{2,}', " "]
```
- if more than 1 occurence of a given punctuation it means that it was used for formatting 


### 6 get math spacing

```python
[r'([><=+%\/:])', r' \1 ']
```
- dont allow operators to stick to characters around them


### 6 get dimension spacing

```python
[r'(\d+[.\d]*)([x])', r'\1 \2 ' ]
```
- dont allow operators to stick to characters around them