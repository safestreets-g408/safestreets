import React from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { NotificationBadge } from '../notifications';

const HeaderRight = ({ onPress }) => {
  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={onPress}>
        <NotificationBadge />
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingRight: 10,
  },
});

export default HeaderRight;
