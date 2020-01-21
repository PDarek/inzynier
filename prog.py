from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time

start_time = time.time()
n_samples = 100
n_features = 20
n_informative = 5
n_redundant = 2
n_repeated = 2

X, y = make_classification(n_samples, n_features, n_informative, n_redundant, n_repeated)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

#y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression: {:.2f}'.format(logreg.score(X_test, y_test)))
#confusion_matrix = confusion_matrix(y_test, y_pred)
#print(confusion_matrix)
#print(classification_report(y_test, y_pred))

"""

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('RO characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()

"""

print("--- %s seconds ---" % (time.time() - start_time))
