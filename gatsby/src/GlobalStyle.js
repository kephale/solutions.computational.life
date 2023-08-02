// src/GlobalStyle.js

import { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  header {
    background-color: ${({ theme }) => theme.colors.background};
    /* You can also add other global styles here */
  }
  body {
    background-color: ${({ theme }) => theme.colors.background};
    /* You can also add other global styles here */
  }
`;

export default GlobalStyle;
