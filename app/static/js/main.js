document.addEventListener('DOMContentLoaded', function() {
    // Check token and redirect to login if not present
    const currentPath = window.location.pathname;
    
    if (currentPath !== '/login' && !localStorage.getItem('access_token')) {
        window.location.href = '/login';
    }
    
    // Add Bootstrap icons
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css';
    document.head.appendChild(link);
});
