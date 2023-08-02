import React from 'react';
import { ThemeProvider } from 'styled-components';
import theme from './gatsby-theme-album';
import GlobalStyle from './GlobalStyle';

export const wrapRootElement = ({ element }) => (
  <ThemeProvider theme={theme}>
    <GlobalStyle />
    {element}
  </ThemeProvider>
);
