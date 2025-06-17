import { getAuthToken } from './auth';
import { API_BASE_URL } from '../config';

// Get task assignments for the current field worker
export const fetchFieldWorkerTasks = async (status) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    let url = `${API_BASE_URL}/fieldworker/tasks`;
    if (status) {
      url += `?status=${status}`;
    }
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch tasks');
    }
    
    return data.tasks;
  } catch (error) {
    console.error('Error fetching tasks:', error);
    throw error;
  }
};

// Update task status
export const updateTaskStatus = async (taskId, status, notes = '') => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/tasks/${taskId}/status`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status, notes }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to update task status');
    }
    
    return data.task;
  } catch (error) {
    console.error('Error updating task status:', error);
    throw error;
  }
};

// Get task details
export const fetchTaskDetails = async (taskId) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/tasks/${taskId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to fetch task details');
    }
    
    return data.task;
  } catch (error) {
    console.error('Error fetching task details:', error);
    throw error;
  }
};

// Add progress update to a task
export const addTaskProgressUpdate = async (taskId, updateData) => {
  try {
    const token = await getAuthToken();
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await fetch(`${API_BASE_URL}/fieldworker/tasks/${taskId}/progress`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updateData),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to add progress update');
    }
    
    return data.task;
  } catch (error) {
    console.error('Error adding progress update:', error);
    throw error;
  }
};
