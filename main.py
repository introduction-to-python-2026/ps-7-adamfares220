import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- שלב 1: טען ובדוק את הנתונים ---
# טעינת מערך הנתונים של איריס
iris_data = load_iris(as_frame=True) # as_frame=True מחזיר DataFrame של pandas

# המרת הנתונים ל-DataFrame לניתוח קל יותר
df = iris_data.frame

# הוספת שמות מינים מובנים לעמודה חדשה לקריאות טובה יותר בגרפים
df['species'] = iris_data.target_names[iris_data.target]

print("Successfully loaded Iris dataset.")
print("\nFirst 5 rows of the data:")
print(df.head())
print("\nData Info:")
df.info()

# --- שלב 2: בחר תכונות וצור היסטוגרמות ---
# בחירת כל התכונות
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# יצירת היסטוגרמות לכל תכונה
df[features].hist(figsize=(12, 10))
plt.suptitle('Histograms of Iris Dataset Features', y=1.02)
#plt.show() # אם אתה מריץ בסביבה ללא ממשק גרפי, השאר את השורה הזו בהערה

# --- שלב 3: צור תרשימי פיזור וקורלציה ---

# יצירת תרשים זוגות (Pair Plot) כדי להראות קורלציות בין כל זוג תכונות
# זהו תרשים הקורלציה המרכזי למשימה זו
sns.pairplot(df, hue='species')
plt.suptitle('Pair Plot of Iris Dataset with Species Hue', y=1.02)
# שמירת הגרף לקובץ PNG
plt.savefig('correlation.png')
print("\nCorrelation scatter plot figure saved as 'correlation.png'")
#plt.show() # אם אתה מריץ בסביבה ללא ממשק גרפי, השאר את השורה הזו בהערה
