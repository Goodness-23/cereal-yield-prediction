import pickle

model = pickle.load(open('../models/best_model.pkl', 'rb'))

print("Model type:", type(model))
print("\nModel details:")
print(model)