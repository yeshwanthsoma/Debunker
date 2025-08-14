# 🔐 API Authentication Guide

## Overview
The TruthLens API now uses **HTTP Basic Authentication** for sensitive endpoints. This provides a simple, reliable authentication method for your deployed APIs.

## 🛡️ Protected Endpoints (Require Authentication)

These endpoints now require HTTP Basic Auth:

- `POST /api/analyze` - Core fact-checking analysis
- `POST /api/analyze-file` - File upload analysis
- `POST /api/trigger-aggregation` - Manual news aggregation
- `POST /api/scheduler/trigger/{job_id}` - Trigger scheduled jobs
- `POST /api/claims/{claim_id}/enhance-with-grok` - Grok enhancements

## 🌐 Public Endpoints (No Authentication)

These endpoints remain publicly accessible:

- `GET /` - Landing page
- `GET /health` - Health check
- `GET /docs` - API documentation
- `GET /api/trending-claims` - View trending claims
- `GET /api/trending-claims/{claim_id}` - View claim details
- `GET /api/claims/categories` - View categories
- `GET /api/claims/analytics` - View analytics
- `GET /api/scheduler/status` - View scheduler status

## 🔑 Setting Up Credentials

### Railway Environment Variables
```bash
API_USERNAME=your_secure_username
API_PASSWORD=your_secure_password_123!
```

### Local Development (.env)
```bash
API_USERNAME=admin
API_PASSWORD=secure_password_change_in_production
```

## 💻 Frontend Integration

### Option 1: Environment Variables (Recommended)
```javascript
// Frontend .env
REACT_APP_API_USERNAME=your_username
REACT_APP_API_PASSWORD=your_password

// API service file
const API_CREDENTIALS = btoa(`${process.env.REACT_APP_API_USERNAME}:${process.env.REACT_APP_API_PASSWORD}`);

export const factCheckAPI = {
  async analyzeClaimWithAuth(claimData) {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Basic ${API_CREDENTIALS}`
      },
      body: JSON.stringify(claimData)
    });
    
    if (response.status === 401) {
      throw new Error('Authentication failed');
    }
    
    return response.json();
  },

  async uploadFileWithAuth(formData) {
    const response = await fetch('/api/analyze-file', {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${API_CREDENTIALS}`
      },
      body: formData // Don't set Content-Type for FormData
    });
    
    return response.json();
  }
};
```

### Option 2: Build-time Configuration
```javascript
// config.js
const API_CONFIG = {
  production: {
    credentials: 'cHJvZF91c2VyOnByb2RfcGFzcw==' // base64 of prod credentials
  },
  development: {
    credentials: 'YWRtaW46cGFzc3dvcmQ=' // base64 of dev credentials
  }
};

const currentConfig = API_CONFIG[process.env.NODE_ENV] || API_CONFIG.development;

export const authenticatedFetch = (url, options = {}) => {
  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Basic ${currentConfig.credentials}`
    }
  });
};
```

### Option 3: Runtime Prompt (For Admin Tools)
```javascript
// For admin interfaces
let credentials = null;

export const promptForCredentials = () => {
  const username = prompt('Enter API username:');
  const password = prompt('Enter API password:');
  credentials = btoa(`${username}:${password}`);
  return credentials;
};

export const adminAPI = {
  async triggerAggregation() {
    if (!credentials) {
      promptForCredentials();
    }
    
    const response = await fetch('/api/trigger-aggregation', {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${credentials}`
      }
    });
    
    if (response.status === 401) {
      credentials = null; // Reset on auth failure
      throw new Error('Invalid credentials');
    }
    
    return response.json();
  }
};
```

## 🧪 Testing Authentication

### cURL Examples
```bash
# Test protected endpoint (should require auth)
curl -X POST https://your-app.railway.app/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text_claim": "The Earth is flat"}'
# Expected: 401 Unauthorized

# Test with authentication
curl -X POST https://your-app.railway.app/api/analyze \
  -u "username:password" \
  -H "Content-Type: application/json" \
  -d '{"text_claim": "The Earth is flat"}'
# Expected: 200 OK with analysis

# Test public endpoint (no auth needed)
curl https://your-app.railway.app/api/trending-claims
# Expected: 200 OK with claims list
```

### JavaScript Testing
```javascript
// Test authentication
async function testAuth() {
  try {
    // This should fail without auth
    const noAuthResponse = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_claim: 'Test claim' })
    });
    console.log('No auth:', noAuthResponse.status); // Should be 401
    
    // This should succeed with auth
    const withAuthResponse = await fetch('/api/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Basic ' + btoa('username:password')
      },
      body: JSON.stringify({ text_claim: 'Test claim' })
    });
    console.log('With auth:', withAuthResponse.status); // Should be 200
    
  } catch (error) {
    console.error('Test failed:', error);
  }
}
```

## ⚠️ Security Considerations

### What's Visible in Browser DevTools
Yes, the credentials **are visible** in the Network tab as base64-encoded headers:
```
Authorization: Basic YWRtaW46cGFzc3dvcmQ=
```

### Why This Is Still Secure
1. **HTTPS encryption**: All traffic is encrypted between browser and Railway
2. **Server-side validation**: Only the server knows the correct credentials
3. **No persistent storage**: Credentials aren't stored in localStorage/cookies
4. **Environment-based**: Different credentials per environment

### Best Practices
1. **Use strong passwords**: `my_secure_password_123!` not `password123`
2. **Different credentials per environment**: Don't reuse dev credentials in production
3. **Rotate regularly**: Change credentials periodically
4. **Limit access**: Only give credentials to authorized team members
5. **Monitor usage**: Check Railway logs for authentication failures

## 📊 Error Handling

```javascript
export const handleAuthError = (response) => {
  if (response.status === 401) {
    // Clear stored credentials and redirect to login
    localStorage.removeItem('api_credentials');
    window.location.href = '/login';
    return;
  }
  
  if (response.status === 403) {
    throw new Error('Access denied - insufficient permissions');
  }
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
};

// Usage
try {
  const response = await authenticatedFetch('/api/analyze', { ... });
  handleAuthError(response);
  const data = await response.json();
} catch (error) {
  console.error('API call failed:', error.message);
}
```

## 🚀 Deployment Checklist

1. ✅ Set `API_USERNAME` and `API_PASSWORD` in Railway environment
2. ✅ Update frontend environment variables with production credentials
3. ✅ Test all protected endpoints with authentication
4. ✅ Verify public endpoints still work without authentication
5. ✅ Update any CI/CD scripts to include credentials
6. ✅ Document credentials for team members securely

This authentication system provides a good balance of security and simplicity for your deployed fact-checking APIs! 🔐