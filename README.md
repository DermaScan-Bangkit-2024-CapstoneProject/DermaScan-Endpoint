# DermaScan Machine Learning Endpoint

###### This repository contains the code for the DermaScan Model Endpoint.

###### DermaScan Documentation - Capstone Project - C242-PR605

# **API Documentation**

## **Base URL**

##### Development

```
http://0.0.0.0:8080
```

##### Production

```
https://model-api2-546175711662.asia-southeast2.run.app
```

---

### **1. Test Endpoint**

**Endpoint**:  
`Get /test`

**Description**:  
Its used for connectivity test

**Request Body**:

```json
{
    "message": "Ini adalah endpoint model"
}
```

**Response**:
status code: <span style="color:#03ab22;">200</span>

```json
{
    "message": "User Successfully Created"
}
```

---

### **2. Predict**

**Endpoint**:  
`POST /predict`

**Description**:  
It is for predicting skin cancer disease.

**Request Body**:

```json
{
    "file": "image"
}
```

**Response**:
status code: <span style="color:#03ab22;">200</span>

```json
{
    "prediction": "string",
    "probability": "number",
    "probability_percentage": "number",
    "probability_percentage_str": "string",
    "cancerous_status": "string"
}
```
