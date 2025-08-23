# 🔧 Frontend Backend Configuration Guide

## 🚀 **Step-by-Step Full-Stack Deployment**

### **Phase 1: Deploy Backend to Railway** 🚂

1. **Visit Railway**: https://railway.app
2. **Sign in with GitHub**
3. **Deploy from GitHub repo**: `sayaksatpathi/cybershield-ai-streamlit`
4. **Get your backend URL**: `https://your-app-name.railway.app`

### **Phase 2: Update Frontend Configuration** 🔧

Your frontend is already configured for production! Just update the Railway URL:

**File**: `frontend/cybershield_working.html`
**Line 550**: Update the API_BASE URL

```javascript
// 🚀 PRODUCTION API Configuration
// ⚠️ UPDATE THIS URL WITH YOUR RAILWAY BACKEND URL ⚠️
const API_BASE = 'https://YOUR-RAILWAY-APP-NAME.railway.app/api';
```

### **Phase 3: Re-deploy to Netlify** 📱

1. **Create new deployment package**:
   ```bash
   cd frontend/
   zip -r netlify-cybershield-production.zip cybershield_working.html styles.css script.js index.html
   ```

2. **Upload to Netlify**: https://app.netlify.com/drop
3. **Get updated URL**: Your existing Netlify app will be updated

---

## 🎯 **Quick Update Template**

**Replace this line** in `cybershield_working.html`:
```javascript
const API_BASE = 'https://cybershield-backend-production.railway.app/api';
```

**With your actual Railway URL**:
```javascript
const API_BASE = 'https://your-actual-app-name.railway.app/api';
```

---

## 🧪 **Testing Your Backend Connection**

After updating and re-deploying:

1. **Visit your Netlify app**: https://rad-donut-a8e264.netlify.app/
2. **Check browser console** (F12) for connection status
3. **Test file upload** functionality
4. **Verify API responses** are working

---

## 📊 **Expected Deployment Flow**

1. ✅ **Railway Backend**: `https://your-app.railway.app` 
2. ✅ **Updated Frontend**: API calls point to Railway
3. ✅ **Netlify Frontend**: `https://rad-donut-a8e264.netlify.app/`
4. ✅ **Full Integration**: Frontend ↔ Backend communication

---

## ⚠️ **Important Notes**

- **CORS**: Already configured in `simple_backend.py`
- **HTTPS**: Required for production (Railway provides this)
- **API Health**: Test `/api/health` endpoint first
- **File Uploads**: Large file support configured

---

## 🚀 **Ready to Deploy?**

Your frontend is now configured for production backend connection!

**Next steps**:
1. Deploy backend to Railway
2. Update the Railway URL in frontend
3. Re-deploy to Netlify
4. Test full-stack functionality

**Your CyberShield AI will be a complete cloud-based application!** 🛡️
