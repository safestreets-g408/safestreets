import React from 'react';
import { View, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

const Screen = ({ 
  children, 
  style, 
  safeAreaStyle,
  edges = ['top', 'left', 'right'],
  useSafeArea = true 
}) => {
  if (useSafeArea) {
    return (
      <SafeAreaView style={[styles.safeArea, safeAreaStyle]} edges={edges}>
        <View style={[styles.screen, style]}>{children}</View>
      </SafeAreaView>
    );
  }
  
  return <View style={[styles.screen, style]}>{children}</View>;
};

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  screen: {
    flex: 1,
  },
});

export default Screen;
